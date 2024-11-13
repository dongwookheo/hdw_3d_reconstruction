import cv2
import numpy as np


K = np.array([[1086, 0, 512], [0, 1086, 384], [0, 0, 1]])
DISTORTION_COEFFICIENT = np.array([-0.0568965, 0, 0, 0, 0])
VISUALIZATION = True


def main():
    # 이미지 읽기
    img1 = cv2.imread("data/003.jpg")
    img2 = cv2.imread("data/005.jpg")

    # 이미지 확인
    if img1 is None or img2 is None:
        print("Error: Image not found or unable to load.")
        return
    else:
        img1_height, img1_width = img1.shape[:2]
        img2_height, img2_width = img2.shape[:2]
        print(f"Image1 height, width: {img1_height}, {img1_width}")
        print(f"Image2 height, width: {img2_height}, {img2_width}")

    # Feature Matching (SIFT)
    sift = cv2.SIFT.create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Visualize matching
    img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None)
    img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None)

    # Feature matching (brute-force)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(
        descriptors1, descriptors2, k=2
    )  # Matching with Euclidean distance

    # Ratio test from SIFT (David Lowe)
    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.8 * m2.distance]

    # Draw matches
    matching_result_bf = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, good_matches, None
    )

    idx_img1 = [match.queryIdx for match in good_matches]
    idx_img2 = [match.trainIdx for match in good_matches]
    keypoints1_matched = np.array([keypoints1[i].pt for i in idx_img1]).reshape(-1, 2)
    keypoints2_matched = np.array([keypoints2[i].pt for i in idx_img2]).reshape(-1, 2)

    # Essential matrix, R, t 계산
    ret, E, R, t, mask = cv2.recoverPose(
        keypoints1_matched,
        keypoints2_matched,
        K,
        DISTORTION_COEFFICIENT,
        K,
        DISTORTION_COEFFICIENT,
        method=cv2.USAC_MAGSAC,
    )

    print("* Essential matrix:", E, sep="\n")
    print("* Rotation matrix:", R, sep="\n")
    print("* Translation vector:", t, sep="\n")

    # 3D reconstruction (triangulation) in normalized image plane
    keypoints1_matched = keypoints1_matched[mask.ravel() == 1]
    keypoints2_matched = keypoints2_matched[mask.ravel() == 1]

    normalized_keypoints1 = cv2.undistortPoints(
        keypoints1_matched, K, DISTORTION_COEFFICIENT
    ).reshape(-1, 2)
    normalized_keypoints2 = cv2.undistortPoints(
        keypoints2_matched, K, DISTORTION_COEFFICIENT
    ).reshape(-1, 2)
    P1 = np.eye(3, 4, dtype=np.float32)
    P2 = np.hstack((R, t))
    #! cv::triangulatePoints(): 모든 arguments는 float type으로 넣어주어야 한다.
    X = cv2.triangulatePoints(
        P1,
        P2,
        normalized_keypoints1.T,
        normalized_keypoints2.T,
    )
    X /= X[3]
    X = X[:3].T

    # P1 = np.eye(3, 4, dtype=np.float64)
    # P2 = K @ np.hstack((R, t))
    # X = cv2.triangulatePoints(P1, P2, keypoints1_matched.T, keypoints2_matched.T)
    # X /= X[3]
    # X = X[:3].T

    if VISUALIZATION:
        import open3d as o3d

        # 3D reconstruction 결과 시각화 (Open3D)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 5  # Increase the point size
        vis.run()
        vis.destroy_window()

        img = np.hstack((img1, img2))
        for k1, k2 in zip(keypoints1_matched, keypoints2_matched):
            k2[0] += img1_width

            k1, k2 = (k1 + 0.5), (k2 + 0.5)
            k1 = k1.astype(np.int32)
            k2 = k2.astype(np.int32)

            # 랜덤한 색깔로 매칭된 점을 연결
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, k1, k2, color, 1)
            cv2.circle(img, k1, 5, color, 1)
            cv2.circle(img, k2, 5, color, 1)

        cv2.imshow("Inlier with RANSAC", img)
        cv2.imshow("Matched images", matching_result_bf)
        cv2.waitKey(0)
        print(len(keypoints1), len(keypoints1_matched))


if __name__ == "__main__":
    main()
