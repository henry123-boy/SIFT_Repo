**姓名：肖宇曦**                                                     **学号：2019302130205**

**选题：SIFT特征匹配**

[TOC]

### 实验数据

| <img src="F:\henry's py\project_all\Sift_Repo\datasets\image2.png" style="zoom:25%;" /> | <img src="F:\henry's py\project_all\Sift_Repo\datasets\image1.png" style="zoom:25%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="F:\henry's py\project_all\Sift_Repo\datasets\image11.png" style="zoom: 33%;" /> | <img src="F:\henry's py\project_all\Sift_Repo\datasets\image12.png" style="zoom: 33%;" /> |



### 基本原理与算法步骤

#### 基本步骤

1. 适当模糊和上采样输入图像生成图像金字塔的基础影像
2. 计算图像金字塔中的层数octaves
3. 创建不同尺度下的高斯核（用来高斯模糊）
4. 在不同尺度下反复模糊基础影像同时下采样影像生成图像金字塔
5. 在每个尺度上相减高斯图像生成高斯差分DoG图像的金字塔
6. 剔除关键点的重复项
7. 生成每个关键点的descriptors，但这里有个很重要的trick每个keypoint是有梯度方向的，然后旋转align算descriptors，要不然会影像匹配效果

#### 关键代码实现

- 设计了SIFT算子类

```python
class SIFT():
    def __init__(self,
                 sigma=1.6,
                 num_intervals=3,
                 assumed_blur=0.5,
                 image_border_width=5):
        '''
        :param sigma: 高斯模糊的模糊参数
        :param num_intervals: 影像层数
        :param assumed_blur: 假定相机模糊的参数
        :param image_border_width: 影像边缘宽度
        '''
        self.sigma=sigma
        self.num_intervals = num_intervals
        self.assumed_blur=assumed_blur
        self.image_border_width=image_border_width
```

- 实现计算关键点和关键点描述子计算

```python
def computeKeypointsAndDescriptors(self,image):
    image = image.astype('float32')
    base_image = self.generateBaseImage(image, self.sigma, self.assumed_blur) 
    #生成基础影像
    num_octaves = self.computeNumberOfOctaves(base_image.shape)
    #计算octaves的层数
    gaussian_kernels = self.generateGaussianKernels(self.sigma, self.num_intervals)
    #生成高斯核
    print("generating the gaussian images")
    #生成高斯影像
    gaussian_images = self.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    print("generating the DOG images")
    #生成DOG影像，做高斯差分
    dog_images = self.generateDoGImages(gaussian_images)
    print("extracting the keypoints")
    #生成关键点
    keypoints = self.findScaleSpaceExtrema(gaussian_images, dog_images, self.num_intervals, self.sigma, self.image_border_width)
    keypoints = utils.removeDuplicateKeypoints(keypoints)
    #将关键点的坐标resize到原始坐标
    keypoints = utils.convertKeypointsToInputImageSize(keypoints)
    #计算描述子
    descriptors = self.generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors
```

- 生成不同的高斯核

  ```python
  def generateGaussianKernels(self,sigma,num_intervals):
      num_images_per_octave = num_intervals + 3
      k = 2 ** (1. / num_intervals)      #每个尺度空间的比例
      gaussian_kernels = np.zeros(
          num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
      gaussian_kernels[0] = sigma
      for image_index in range(1, num_images_per_octave):
          sigma_previous = (k ** (image_index - 1)) * sigma
          sigma_total = k * sigma_previous
          gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)    #sift论文里面提到了因为相机初始具有一个sigma模糊，所以要剪掉
          return gaussian_kernels
  
  ```

- 生成高斯影像

  ```python
  def generateGaussianImages(self,image, num_octaves, gaussian_kernels):
      """Generate scale-space pyramid of Gaussian images
          """
      gaussian_images = []
  
      for octave_index in range(num_octaves):
          gaussian_images_in_octave = []
          gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
          for gaussian_kernel in gaussian_kernels[1:]:
              image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
              gaussian_images_in_octave.append(image)
              gaussian_images.append(gaussian_images_in_octave)
              octave_base = gaussian_images_in_octave[-3]
              image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                                 interpolation=cv2.INTER_NEAREST)
              return np.array(gaussian_images)
  ```

- 生成DoG

  ```python
  def generateDoGImages(self,gaussian_images):
      """Generate Difference-of-Gaussians image pyramid
          """
      dog_images = []
  
      for gaussian_images_in_octave in gaussian_images:
          dog_images_in_octave = []
          for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
              dog_images_in_octave.append(np.subtract(second_image,
                                                      first_image))
              #将相邻的高斯影像做subtract相减
              dog_images.append(dog_images_in_octave)
              return np.array(dog_images)
  ```

  

- 找到每个尺度下的极值点（Coarse的关键点）

  ```python
  def findScaleSpaceExtrema(self,gaussian_images, dog_images, num_intervals, sigma, image_border_width,contrast_threshold=0.04):
      threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
      keypoints = []
      for octave_index, dog_images_in_octave in enumerate(dog_images):
          for image_index, (first_image, second_image, third_image) in enumerate(
              zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
              # (i, j) is the center of the 3x3 array
              for i in range(image_border_width, first_image.shape[0] - image_border_width):
                  for j in range(image_border_width, first_image.shape[1] - image_border_width):
                      if utils.isPixelAnExtremum(first_image[i - 1:i + 2, j - 1:j + 2],second_image[i - 1:i + 2, j - 1:j + 2],third_image[i - 1:i + 2, j - 1:j + 2], threshold):
                          localization_result = self.localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index,  num_intervals, dog_images_in_octave,sigma, contrast_threshold,image_border_width)
                          if localization_result is not None:
                              keypoint, localized_image_index = localization_result
                              keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave_index,gaussian_images[octave_index][localized_image_index])
                    for keypoint_with_orientation in keypoints_with_orientations:
                                  keypoints.append(keypoint_with_orientation)
                                  return keypoints
  ```

- 生成描述子

  ```python
  def generateDescriptors(self,keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
      """Generate descriptors for each keypoint
          """
      logger.debug('Generating descriptors...')
      descriptors = []
  
      for keypoint in keypoints:
          octave, layer, scale = utils.unpackOctave(keypoint)
          gaussian_image = gaussian_images[octave + 1, layer]
          num_rows, num_cols = gaussian_image.shape
          point = (scale * np.array(keypoint.pt)).astype('int')
          bins_per_degree = num_bins / 360.
          angle = 360. - keypoint.angle
          cos_angle = np.cos(np.deg2rad(angle))
          sin_angle = np.sin(np.deg2rad(angle))
          weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
          row_bin_list = []
          col_bin_list = []
          magnitude_list = []
          orientation_bin_list = []
          histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects
  
          # Descriptor window size (described by half_width) follows OpenCV convention
          hist_width = scale_multiplier * 0.5 * scale * keypoint.size
          half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
          half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image
  
          for row in range(-half_width, half_width + 1):
              for col in range(-half_width, half_width + 1):
                  row_rot = col * sin_angle + row * cos_angle
                  col_rot = col * cos_angle - row * sin_angle
                  row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                  col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                  if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                      window_row = int(np.round(point[1] + row))
                      window_col = int(np.round(point[0] + col))
                      if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                          dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                          dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                          gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                          gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                          weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                          row_bin_list.append(row_bin)
                          col_bin_list.append(col_bin)
                          magnitude_list.append(weight * gradient_magnitude)
                          orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
  
                          for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                              # Smoothing via trilinear interpolation
                              # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                              # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                              row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                              row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                              if orientation_bin_floor < 0:
                                  orientation_bin_floor += num_bins
                                  if orientation_bin_floor >= num_bins:
                                      orientation_bin_floor -= num_bins
  
                                      c1 = magnitude * row_fraction
                                      c0 = magnitude * (1 - row_fraction)
                                      c11 = c1 * col_fraction
                                      c10 = c1 * (1 - col_fraction)
                                      c01 = c0 * col_fraction
                                      c00 = c0 * (1 - col_fraction)
                                      c111 = c11 * orientation_fraction
                                      c110 = c11 * (1 - orientation_fraction)
                                      c101 = c10 * orientation_fraction
                                      c100 = c10 * (1 - orientation_fraction)
                                      c011 = c01 * orientation_fraction
                                      c010 = c01 * (1 - orientation_fraction)
                                      c001 = c00 * orientation_fraction
                                      c000 = c00 * (1 - orientation_fraction)
  
                                      histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                                      histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                                      histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                                      histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                                      histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                                      histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                                      histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                                      histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111
  
                                      descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
                                      # Threshold and normalize descriptor_vector
                                      threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
                                      descriptor_vector[descriptor_vector > threshold] = threshold
                                      descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
                                      # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
                                      descriptor_vector = np.round(512 * descriptor_vector)
                                      descriptor_vector[descriptor_vector < 0] = 0
                                      descriptor_vector[descriptor_vector > 255] = 255
                                      descriptors.append(descriptor_vector)
                                      return np.array(descriptors, dtype='float32')
  ```

### 实验结果展示

- 匹配结果

  | <img src="F:\henry's py\project_all\Sift_Repo\datasets\result1.png" alt="image-20220430113813445" style="zoom:50%;" /> |      |
  | ------------------------------------------------------------ | ---- |

  

### 注：

代码已上传至一下Repo中：

https://github.com/henry123-boy/SIFT_Repo









