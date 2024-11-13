# hdw_3d_reconstruction
- 3D reconstruction을 위한 레포지토리
- 예제 이미지는 Oxford university의 multi-view data/Wadham College dataset 중 3번, 5번 이미지를 사용하였습니다.
- **Camera matrix *K*** 와 **distortion coefficient**는 사전에 주어졌다고 가정하였습니다.
- 최적화 문제 이전까지의 과정을 진행하였습니다.
- 아래의 절차 순으로 진행하였습니다.
  1. SIFT를 이용해서 keypoints와 descriptors를 계산합니다.
  2. Brute-force 방법으로 매칭을 수행한 뒤, 첫 번째로 좋은 매칭과 두 번째로 좋은 매칭 사이의 Euclidean distance를 비교하여 명확히 구분되는 경우를 좋은 매칭점으로 남깁니다.
  3. Keypoints 쌍과 ***K***, **distortion coefficient**를 이용하여 **Essential matrix *E***, **Relative camera pose *R, t*** 를 계산합니다.
  4. Keypoints 쌍을 normalized image plane으로 변환합니다.
  5. ***R, t*** 를 이용하여 world coordinate에서 normalized image plane으로의 **projection matrix *P*** 를 계산합니다.
  6. ***P*** 와 normalized image plane으로 변환된 keypoints를 이용하여 **3D points**를 계산합니다.

## Results
### 1. 2D point visualization on two images
![matching_result](/resource/matching_result.png)
![inlier_after_ransac](/resource/inlier_after_ransac.png)
### 2. The essential matrix *E* between two images
![e_result](/resource/e_result.png)
### 3. The rotation and translation, *R* and *t*
![rt_result](/resource/rt_result.png)
### 4. 3D visualization of reconstructed 3D points in two viewpoints
![3d_recon_result](/resource/3d_recon_result.png)
- Points의 크기를 5으로 설정한 이미지입니다.

## Discussion
- Brute-force matching 없이, 최대한 많은 매칭 결과를 활용하여 `RANSAC`을 적용하려고 했을 때 잘 되지 않았습니다.
  - `SIFT`를 통해 뽑은 keypoints가 5957쌍인데, [inlier after RANSAC](#inlier_after_ransac) 결과에서 볼 수 있듯이 inlier의 수가 현저히 적습니다 (153쌍).
  - 따라서, `RANSAC`이 잘 동작하지 않았습니다. (최소 20~30% 이상인 경우 동작)

## Todo
- [ ] Feature matching 방법 바꿔보기 (RoMa, XFeat, ...)
- [ ] reprojection error를 최소화하는 과정까지 진행해보기
