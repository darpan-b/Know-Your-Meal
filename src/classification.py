import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from torchvision import models, transforms
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis


class PEClassifier:
    def __init__(
        self,
        pe_model,
        preprocess,
        sim_thr,
        device,
        num_patches: int = 5,
        patch_size: int = 64,
        sim_method: str = 'cosine',  # 'cosine','euclidean','manhattan','mahalanobis','knn'
        knn_k: int = 5
    ):
        self.pe_model = pe_model
        self.preprocess = preprocess
        self.sim_thr = sim_thr
        self.device = device
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.sim_method = sim_method
        self.knn_k = knn_k

    def _build_distance_helpers(self, proto_embs: dict[str, np.ndarray]):
        """
        Builds distance helper structures (covariance, KNN index) based on the
        prototype embeddings provided. Handles both single (D,) and multiple (N, D)
        prototype shapes per class.
        """
        self._proto_embs = proto_embs
        self._classes = list(proto_embs.keys())

        # Determine if we have single (D,) or multiple (N, D) prototypes per class
        # Assumes all classes have the same prototype structure (either all single or all multiple)
        first_class_proto = next(iter(proto_embs.values()))
        is_multi_prototype = (first_class_proto.ndim == 2)

        if is_multi_prototype:
            # Aggregate all individual prototype embeddings from all classes
            all_individual_feats = []
            self._prototype_class_labels = []
            for cls, embeddings in proto_embs.items():
                # embeddings shape is (N, D)
                all_individual_feats.extend(embeddings)
                self._prototype_class_labels.extend(
                    [cls] * embeddings.shape[0])

            all_individual_feats = np.stack(
                all_individual_feats, axis=0)  # Shape (TotalPrototypes, D)

            if self.sim_method == 'mahalanobis':
                # Build covariance matrix on all individual prototype embeddings
                cov = np.cov(all_individual_feats, rowvar=False)
                # Add small diagonal for numerical stability, especially if TotalPrototypes < D
                cov += np.eye(cov.shape[0]) * 1e-6
                self._inv_cov = np.linalg.inv(cov)

            elif self.sim_method == 'knn':
                # Build KNN index on all individual prototype embeddings
                # Note: NearestNeighbors works with distances. Cosine similarity -> Cosine distance
                self._knn = NearestNeighbors(
                    n_neighbors=self.knn_k, metric='cosine'
                ).fit(all_individual_feats)

        else:  # Single prototype per class (ndim == 1)
            if self.sim_method == 'mahalanobis':
                # Build covariance matrix on the single prototypes from all classes
                # Shape (NumClasses, D)
                all_mean_feats = np.stack(list(proto_embs.values()), axis=0)
                cov = np.cov(all_mean_feats, rowvar=False)
                # Add small diagonal for numerical stability, especially if NumClasses < D
                cov += np.eye(cov.shape[0]) * 1e-6
                self._inv_cov = np.linalg.inv(cov)

            elif self.sim_method == 'knn':
                # Build KNN index on the single prototypes from all classes
                # Shape (NumClasses, D)
                all_mean_feats = np.stack(list(proto_embs.values()), axis=0)
                # _classes is already set
                self._knn = NearestNeighbors(
                    n_neighbors=min(self.knn_k, len(self._classes)), metric='cosine'
                ).fit(all_mean_feats)

    def _score(self, feat: np.ndarray):
        """
        Calculates similarity scores between a query feature vector (feat) and all
        prototype embeddings. Returns the predicted class and the best score.
        Handles both single and multiple prototypes per class.

        Args:
            feat (np.ndarray): The query embedding (shape D,).

        Returns:
            tuple[str, float]: (predicted_class, best_score)
        """
        pe = self.sim_method
        # The dictionary of prototype embeddings {class: np.ndarray}
        P = self._proto_embs

        # Determine prototype structure if P is not empty
        if not P:
            return None, 0.0  # No prototypes to score against

        first_proto_array = next(iter(P.values()))
        is_multi_prototype = (first_proto_array.ndim == 2)

        if is_multi_prototype:
            # --- Scoring against multiple prototypes per class (N, D) ---
            best_cls = None
            best_score = -float('inf')  # Use negative infinity for comparison

            if pe == 'knn':
                # KNN is handled differently as it's a global neighbor search
                # Query the global KNN index (built on all individual prototypes)
                dist, idx = self._knn.kneighbors(
                    feat[None], return_distance=True)
                dist, idx = dist[0], idx[0]  # Shape (k,), (k,)

                # Get the class labels for the K nearest neighbors
                neighbor_classes = [
                    self._prototype_class_labels[i] for i in idx]

                # Majority vote among the K neighbors
                if not neighbor_classes:
                    return None, 0.0  # Should not happen if k > 0 and index was built
                majority_cls = Counter(neighbor_classes).most_common(1)[0][0]

                # Calculate average similarity (1 - distance) for neighbors belonging to the majority class
                sims = 1.0 - dist  # Convert distances to similarities
                scores_for_majority_cls = [s for c, s in zip(
                    neighbor_classes, sims) if c == majority_cls]
                avg_sim = float(np.mean(scores_for_majority_cls)
                                ) if scores_for_majority_cls else 0.0

                # The 'score' returned for KNN is the average similarity of the majority neighbors
                # The 'threshold' should likely be applied to this average similarity
                return majority_cls, avg_sim

            # For other methods, iterate through each class and find the best match within that class
            for cls, class_prototypes in P.items():  # class_prototypes shape is (N, D)
                # Calculate scores between the query feat (1, D) and all prototypes in the class (N, D)
                if pe == 'cosine' or pe not in ('euclidean', 'manhattan', 'mahalanobis', 'clip'):
                    # cosine_similarity returns (1, N) matrix, take the first row (N,)
                    sims = cosine_similarity(feat[None], class_prototypes)[0]
                    # Max similarity to any prototype in this class
                    score_for_this_class = np.max(sims)
                elif pe == 'euclidean':
                    # euclidean_distances returns (1, N) matrix, take the first row (N,)
                    dists = euclidean_distances(
                        feat[None], class_prototypes)[0]
                    # Convert distance to similarity: 1 / (1 + distance)
                    sims = 1.0 / (1.0 + dists)
                    # Max similarity (min distance)
                    score_for_this_class = np.max(sims)
                elif pe == 'manhattan':
                    # manhattan_distances returns (1, N) matrix, take the first row (N,)
                    dists = manhattan_distances(
                        feat[None], class_prototypes)[0]
                    # Convert distance to similarity: 1 / (1 + distance)
                    sims = 1.0 / (1.0 + dists)
                    # Max similarity (min distance)
                    score_for_this_class = np.max(sims)
                elif pe == 'mahalanobis':
                    # Calculate mahalanobis distance between feat and each prototype in the class
                    # Need to iterate or use a potentially less efficient broadcast calculation
                    # More efficient: Iterate through prototypes in the class
                    mahalanobis_dists = [mahalanobis(
                        feat, p, self._inv_cov) for p in class_prototypes]
                    # Convert distance to similarity: 1 / (1 + distance)
                    sims = 1.0 / (1.0 + np.array(mahalanobis_dists))
                    # Max similarity (min distance)
                    score_for_this_class = np.max(sims)
                # elif pe == 'clip': # Add CLIP specific scoring if needed
                #     pass
                else:
                    raise ValueError(f"Unknown sim_method={pe}")

                # Update overall best score and class
                if score_for_this_class > best_score:
                    best_score = score_for_this_class
                    best_cls = cls

            return best_cls, best_score

        else:  # --- Scoring against single prototype per class (D,) ---
            # This is the original logic
            if pe == 'cosine' or pe not in ('euclidean', 'manhattan', 'mahalanobis', 'knn', 'clip'):
                sims = {c: float(cosine_similarity(
                    feat[None], P[c][None])[0, 0]) for c in P}
                return max(sims.items(), key=lambda kv: kv[1])

            if pe == 'euclidean':
                # Calculate Euclidean distance and convert to similarity
                # Using euclidean_distances for potential slight performance benefit over np.linalg.norm loop
                sims = {}
                for c, proto in P.items():
                    dist = euclidean_distances(feat[None], proto[None])[0, 0]
                    sims[c] = 1.0 / (1.0 + dist)
                return max(sims.items(), key=lambda kv: kv[1])

            if pe == 'manhattan':
                # Calculate Manhattan distance and convert to similarity
                sims = {}
                for c, proto in P.items():
                    dist = manhattan_distances(feat[None], proto[None])[0, 0]
                    sims[c] = 1.0 / (1.0 + dist)
                return max(sims.items(), key=lambda kv: kv[1])

            if pe == 'mahalanobis':
                # Calculate Mahalanobis distance and convert to similarity
                sims = {}
                for c, proto in P.items():
                    dist = mahalanobis(feat, proto, self._inv_cov)
                    sims[c] = 1.0 / (1.0 + dist)
                return max(sims.items(), key=lambda kv: kv[1])

            if pe == 'knn':
                # Query the global KNN index (built on single prototypes)
                all_mean_feats = np.stack(list(P.values()), axis=0)
                # returns cosine-distance, so sim = 1 - dist
                dist, idx = self._knn.kneighbors(
                    feat[None], return_distance=True)
                dist, idx = dist[0], idx[0]  # Shape (k,), (k,)

                # Get the class labels for the K nearest neighbors (using self._classes which has unique class names)
                neighbor_classes = [self._classes[i] for i in idx]

                # Majority vote among the k prototypes
                if not neighbor_classes:
                    return None, 0.0
                majority_cls = Counter(neighbor_classes).most_common(1)[0][0]

                # Average sim among prototypes of that class (among the K neighbors)
                sims = 1.0 - dist  # Convert distances to similarities
                scores_for_majority_cls = [s for c, s in zip(
                    neighbor_classes, sims) if c == majority_cls]
                avg_sim = float(np.mean(scores_for_majority_cls)
                                ) if scores_for_majority_cls else 0.0

                # The 'score' returned for KNN is the average similarity of the majority neighbors
                return majority_cls, avg_sim

            # (you could add a 'clip' branch here if you wire in CLIPZeroShotClassifier)
            raise ValueError(f"Unknown sim_method={pe}")

    # def classify(self, masks: np.ndarray, img_rgb: np.ndarray, proto_embs: dict[str, np.ndarray]):
    #     """
    #     Classifies regions defined by masks in an image using prototype embeddings.

    #     Args:
    #         masks (np.ndarray): Boolean masks for regions of interest. Can be (H, W)
    #                             for a single mask or (N_masks, H, W) for multiple masks.
    #         img_rgb (np.ndarray): The input image in RGB format (H, W, 3).
    #         proto_embs (dict[str, np.ndarray]): Dictionary mapping class names (str)
    #                                             to prototype embeddings (np.ndarray).
    #                                             Embeddings can be single (D,) or multiple (N, D).

    #     Returns:
    #         tuple[dict[str, np.ndarray], dict[str, float]]:
    #             - class_masks: Dictionary mapping class names to combined boolean masks.
    #             - confs: Dictionary mapping class names to the highest confidence score
    #                     assigned to any region classified as that class.
    #     """
    #     H, W = img_rgb.shape[:2]
    #     class_masks = {}  # Only add classes when predicted
    #     confs = {}        # Store highest confidence per predicted class

    #     # Prepare helpers (covariance, knn index, etc.)
    #     try:
    #         self._build_distance_helpers(proto_embs)
    #     except Exception as e:
    #         print(f"Error building distance helpers: {e}")
    #         return class_masks, confs

    #     # Ensure masks is 3D
    #     masks_to_process = masks if masks.ndim > 2 else masks[None, :, :]

    #     for mask in masks_to_process:
    #         ys, xs = mask.nonzero()
    #         if ys.size == 0:
    #             continue

    #         votes = []
    #         sims_by_class = {c: [] for c in proto_embs}

    #         for _ in range(self.num_patches):
    #             idx = np.random.randint(0, len(ys))
    #             cy, cx = int(ys[idx]), int(xs[idx])
    #             half = self.patch_size // 2
    #             y0 = max(0, cy - half)
    #             x0 = max(0, cx - half)
    #             if y0 + self.patch_size > H:
    #                 y0 = H - self.patch_size
    #             if x0 + self.patch_size > W:
    #                 x0 = W - self.patch_size
    #             y0, x0 = max(0, y0), max(0, x0)

    #             patch = img_rgb[y0:y0 + self.patch_size,
    #                             x0:x0 + self.patch_size]
    #             patch_pil = Image.fromarray(patch)

    #             try:
    #                 img_t = self.preprocess(
    #                     patch_pil).unsqueeze(0).to(self.device)
    #                 with torch.no_grad():
    #                     if hasattr(self.pe_model, 'encode_image'):
    #                         feat = self.pe_model.encode_image(img_t)
    #                     elif hasattr(self.pe_model, 'forward_features'):
    #                         feat = self.pe_model.forward_features(img_t)
    #                     else:
    #                         raise AttributeError(
    #                             "Model does not have encode_image or forward_features.")

    #                 feat = feat.float().cpu().numpy().reshape(-1)
    #                 feat /= (np.linalg.norm(feat) + 1e-8)

    #                 pred_cls, score = self._score(feat)

    #                 if pred_cls is not None:
    #                     votes.append(pred_cls)
    #                     sims_by_class[pred_cls].append(score)

    #             except Exception as e:
    #                 print(
    #                     f"    ⚠️ Error processing patch at ({cy}, {cx}): {e}")
    #                 continue

    #         if not votes:
    #             continue

    #         vote_counts = Counter(votes)
    #         majority_cls, count = vote_counts.most_common(1)[0]
    #         avg_sim = float(np.mean(sims_by_class.get(majority_cls, [0.0])))

    #         if majority_cls not in class_masks:
    #             class_masks[majority_cls] = np.zeros((H, W), dtype=bool)
    #         class_masks[majority_cls] |= mask
    #         confs[majority_cls] = max(confs.get(majority_cls, 0.0), avg_sim)

    #     return class_masks, confs
    def classify(self, masks: np.ndarray, img_rgb: np.ndarray, proto_embs: dict[str, np.ndarray]):
        """
        Classifies regions defined by masks in an image using prototype embeddings.

        Args:
            masks (np.ndarray): Boolean masks for regions of interest. Can be (H, W)
                                for a single mask or (N_masks, H, W) for multiple masks.
            img_rgb (np.ndarray): The input image in RGB format (H, W, 3).
            proto_embs (dict[str, np.ndarray]): Dictionary mapping class names (str)
                                                to prototype embeddings (np.ndarray).
                                                Embeddings can be single (D,) or multiple (N, D).

        Returns:
            tuple[dict[str, np.ndarray], dict[str, float]]:
                - class_masks: Dictionary mapping class names to combined boolean masks.
                - confs: Dictionary mapping class names to the highest confidence score
                        assigned to any region classified as that class.
        """
        H, W = img_rgb.shape[:2]
        # Initialize class masks and confidence scores
        class_masks = {c: np.zeros((H, W), bool) for c in proto_embs}
        confs = {}  # Stores the highest confidence score for each predicted class

        # Prepare any helpers (covariance, knn index) based on the loaded prototypes
        try:
            self._build_distance_helpers(proto_embs)
        except Exception as e:
            print(f"Error building distance helpers: {e}")
            return class_masks, confs  # Return empty if helpers fail

        # Ensure masks is 3D even if only one mask is provided
        masks_to_process = masks if masks.ndim > 2 else masks[None, :, :]

        # Iterate through each individual mask (region of interest)
        for mask in masks_to_process:
            ys, xs = mask.nonzero()  # Get coordinates of pixels within the mask
            if ys.size == 0:
                continue  # Skip empty masks

            # --- Patch-based classification for this region ---
            votes = []  # Store predicted class for each patch
            # Store scores for each patch, categorized by predicted class
            sims_by_class = {c: [] for c in proto_embs}

            # Sample and classify multiple patches within the current mask
            for _ in range(self.num_patches):
                # Sample a random point within the mask
                idx = np.random.randint(0, len(ys))
                cy, cx = int(ys[idx]), int(xs[idx])

                # Extract square patch centered at (cy, cx)
                half = self.patch_size // 2
                y0 = max(0, cy - half)
                x0 = max(0, cx - half)
                # Adjust coordinates to ensure patch fits entirely within the image
                if y0 + self.patch_size > H:
                    y0 = H - self.patch_size
                if x0 + self.patch_size > W:
                    x0 = W - self.patch_size
                # Ensure coordinates are not negative
                y0, x0 = max(0, y0), max(0, x0)

                # Extract the patch from the image
                patch = img_rgb[y0:y0 + self.patch_size,
                                x0:x0 + self.patch_size]
                patch_pil = Image.fromarray(patch)

                # Preprocess and encode the patch
                try:
                    img_t = self.preprocess(
                        patch_pil).unsqueeze(0).to(self.device)
                    # with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    with torch.no_grad():
                        # Ensure model has encode_image or forward_features
                        if hasattr(self.pe_model, 'encode_image'):
                            feat = self.pe_model.encode_image(img_t)
                        elif hasattr(self.pe_model, 'forward_features'):
                            feat = self.pe_model.forward_features(img_t)
                        else:
                            raise AttributeError(
                                "Model does not have encode_image or forward_features method.")

                    feat = feat.float().cpu().numpy().reshape(-1)
                    # Normalize the query patch embedding
                    feat /= (np.linalg.norm(feat) + 1e-8)

                    # Score the patch embedding against prototypes
                    pred_cls, score = self._score(feat)

                    if pred_cls is not None:  # Only add vote if scoring was successful
                        votes.append(pred_cls)
                        sims_by_class[pred_cls].append(score)

                except Exception as e:
                    # Log errors encountered during patch processing or scoring
                    print(
                        f"    ⚠️ Error processing patch at ({cy}, {cx}): {e}")
                    continue  # Skip this patch, try the next one

            # --- Final Classification for this region based on patch votes ---
            if not votes:
                # No patches successfully processed/scored for this region
                continue

            # Perform majority vote on the predicted classes from all patches
            vote_counts = Counter(votes)
            majority_cls, count = vote_counts.most_common(1)[0]

            # Calculate average similarity among the patches that voted for the majority class
            # Use scores collected during patch scoring
            # Default to 0.0 if no scores for majority class
            avg_sim = float(np.mean(sims_by_class.get(majority_cls, [0.0])))

            # Assign the region (mask) to the majority class
            # Decide whether to apply a confidence threshold here or just assign always
            # The original code assigned always, let's stick to that unless sim_thr is intended
            # to filter assignments (which it wasn't in the last version).
            # If you want to filter based on sim_thr:
            # if avg_sim >= self.sim_thr:
            #     class_masks[majority_cls] |= mask
            #     confs[majority_cls] = max(confs.get(majority_cls, 0.0), avg_sim)
            # Else (assign always):
            class_masks[majority_cls] |= mask
            # Store the highest average patch similarity for this class across all regions
            confs[majority_cls] = max(confs.get(majority_cls, 0.0), avg_sim)

        return class_masks, confs


class Resnet101Classifier:
    def __init__(self, sim_thr, device, num_classes=0, checkpoint=None):
        self.model, self.preprocess = self.build_encoder(
            device, checkpoint, num_classes)
        self.model.eval()
        self.sim_thr = sim_thr
        self.device = device

    def build_encoder(device, checkpoint=None, num_classes=0):
        if checkpoint is None:
            m = models.resnet101(pretrained=True)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)

        else:
            m = models.resnet101(pretrained=False)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            m.load_state_dict(torch.load(checkpoint, map_location=device))

        enc = torch.nn.Sequential(*list(m.children())[:-1]).to(device).eval()
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        return enc, tf

    def classify(self, masks: np.ndarray, img_rgb: np.ndarray, proto_embs: dict[str, np.ndarray]):
        H, W = img_rgb.shape[:2]
        class_masks = {c: np.zeros((H, W), bool) for c in proto_embs}
        confs = {}
        for i in range(masks.shape[0] if masks.ndim > 2 else 1):
            mask = masks[i] if masks.ndim > 2 else masks
            ys, xs = mask.nonzero()
            if ys.size == 0:
                continue
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            crop = img_rgb[y0:y1, x0:x1]
            crop_pil = Image.fromarray(crop)

            img_t = self.preprocess(crop_pil).unsqueeze(0)
            feat = self.model(img_t.to(self.device)).squeeze().cpu().numpy()

            feat = feat.float().cpu().numpy().reshape(-1)
            feat /= (np.linalg.norm(feat) + 1e-8)
            sims = {cls: float(cosine_similarity(feat[None], proto_embs[cls][None])[0, 0])
                    for cls in proto_embs}
            pred, best = max(sims.items(), key=lambda kv: kv[1])
            if best >= self.sim_thr:
                class_masks[pred] |= mask
                confs[pred] = max(confs.get(pred, 0.0), best)
        return class_masks, confs
