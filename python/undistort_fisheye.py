import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import sys
import glob

def compute_fov(intrinsic_matrix, image_width, image_height):
    """Computes the camera field of view from the intrinsic parameters and the image dimensions.

    Args:
        intrinsic_matrix: The camera intrinsic matrix.
        image_width: The width of the image.
        image_height: The height of the image.

    Returns:
        The camera field of view in degrees.
    """

    # Get the focal length from the intrinsic matrix.
    focal_length = intrinsic_matrix[0, 0]

    # Compute the horizontal and vertical field of views.
    horizontal_fov = 2 * np.arctan(image_width / (2 * focal_length)) * 180 / np.pi
    vertical_fov = 2 * np.arctan(image_height / (2 * focal_length)) * 180 / np.pi

    # Return the field of view with the larger angle.
    return max(horizontal_fov, vertical_fov)


# model: opencv_fisheye
cha_dim  = [1280, 960]
cha_cent = [6.380461e+02, 4.816096e+02]
cha_foca = [4.914632e+02, 4.908321e+02]
cha_dist = np.array([3.576685e-02,7.124293e-02,-5.729094e-02,1.617010e-02])
cha_K    = np.array([[491.1463, 0.0,      638.0461],
                    [0.0,      490.8321, 481.6096],
                    [0.0,      0.0,      1.0]])
max_fov  = 100.0
print("cha fov: %d " % compute_fov(cha_K, cha_dim[0], cha_dim[1]))

# You should replace these 3 lines with the output in calibration step
DIM=(1280, 800)
K=np.array([[253.90878686798118, 0.0, 649.0303200714142], [0.0, 253.62581932708252, 392.4440213923735], [0.0, 0.0, 1.0]])
# D=np.array([[-0.03116257372965417], [0.008779678117523369], [-0.008619236764532275], [0.0025582074011755674]])
D=np.array([-0.03116257372965417, 0.008779678117523369, -0.008619236764532275, 0.0025582074011755674])
print("Ori fov: %d " % compute_fov(K, DIM[0], DIM[1]))

def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(str(img_path))
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    # print(new_K)
    new_K[0, 0] = 253.0
    new_K[1, 1] = 253.0
    print(new_K)
    print("Ori fov: %d & focal: %f" % (compute_fov(new_K, DIM[0], DIM[1]), new_K[0, 0]))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # undistorted_img = cv2.fisheye.undistortImage(img, K, D=D, Knew=new_K)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return undistorted_img

def cha_undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(str(img_path))
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == cha_dim[0]/cha_dim[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = cha_K * dim1[0] / cha_dim[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, cha_dist, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, cha_dist, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    print(new_K)
    print("cha fov: %d " % compute_fov(new_K, cha_dim[0], cha_dim[1]))
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # undistorted_img = cv2.fisheye.undistortImage(img, K, D=D, Knew=new_K)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return undistorted_img


if __name__ == '__main__':
    from pathlib import Path

    images = glob.glob('/root/e2calib/e2calib/python/frames/e2calib/final/*.png')
    for img_fn in images:
        img_fn = Path(img_fn)
        out_fn = img_fn.parent / (img_fn.stem + '_undist.png')
        print(img_fn)
        print(out_fn)
        out_img = undistort(img_fn)
        cv2.imwrite(str(out_fn), out_img)

    cha_images = glob.glob('/data/datasets/dataset_celepixel/000*.png')
    for img_fn in cha_images:
        img_fn = Path(img_fn)
        out_fn = img_fn.parent / (img_fn.stem + '_undist.png')
        print(img_fn)
        print(out_fn)
        out_img = cha_undistort(img_fn)
        cv2.imwrite(str(out_fn), out_img)
