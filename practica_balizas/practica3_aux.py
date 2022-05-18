import argparse
import apriltag
import cv2
import os
import time
import numpy as np


def read_video(video_path):
    """
    Read video
    :param video_path: path of the mp4
    :return: video
    """
    # Read video
    video = cv2.VideoCapture(video_path)

    # Get the size of the video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the size of the image
    image_size = (width, height)

    return video, image_size


def get_corners_apriltag(frame, tag_family) -> (list, np.array):
    """
    Get image corners of apriltag
    :param frame: image
    :return: image interesting points, corners of the apriltag [left-top, right-top, right-bottom, left-bottom]
    """
    def find_corners(image, debug=False) -> np.array:
        """
        Find corners of the apriltag
        :param image: image
        :return: corners
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        # Remove dimension
        corners = np.squeeze(corners)
        corners = np.int0(corners)

        # Draw corners
        if debug:
            for i, corner in enumerate(corners):
                x, y = corner.ravel()
                cv2.circle(image, (x, y), 3, 255, -1)
                cv2.putText(image, str(i + 1), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

        return corners

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create apriltag detector
    detector = apriltag.Detector(
        apriltag.DetectorOptions(families=tag_family))
    # Detect apriltag
    tags = detector.detect(gray)
    # check if there is a apriltag and get first for calibration
    print(f"{len(tags)} tags found")
    if len(tags) == 0:
        return None, None
    tag = tags[0]
    # Mask tag
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.int32(tag.corners)], (255, 255, 255))
    # Get image masked
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    # Get corners inside masked image
    corners = find_corners(masked_image)
    # Append tag.corners to corners
    corners = np.append(corners, tag.corners, axis=0)

    return corners, np.int32(tag.corners)


def get_object_points_from_image_points(points: np.array, corners_tag: np.array, tag_size) -> np.array:
    """
    Get object point from image point with transformation matrix
    :param points: image points
    :param corners_tag: corners of the apriltag
    :param frame: image
    :return: object point
    """
    # Corners in scene coordinates
    corners_scene = np.array(
        [[0, 0], [0, tag_size], [tag_size, tag_size], [tag_size, 0]])
    # Get transformation matrix
    M = cv2.getPerspectiveTransform(corners_tag.astype(
        np.float32), corners_scene.astype(np.float32))
    # Transform points
    points_scene = cv2.perspectiveTransform(
        points.astype(np.float32)[np.newaxis, ...], M)
    # Remove the first dimension
    points_scene = np.squeeze(points_scene)
    # Round points
    points_scene = np.round(points_scene)
    # Convert points_scene to 3d points with z = 0
    points_scene = np.array([points_scene[:, 0], points_scene[:, 1], np.zeros(
        len(points_scene))], dtype=np.uint32).T
    return points_scene


def draw_image_points_and_object_points(image, image_points, object_points):
    """
    Draw image points and object points
    :param image: image
    :param image_points: image points
    :param object_points: object points
    :param debug: debug
    :return: image
    """
    scale_factor = 12
    # Make a copy of the image
    image_copy = image.copy()
    # Get width of the image
    width = image_copy.shape[1]
    # Draw 3d points along 2d points
    max_x = max(object_points[:, 0])
    # Create a black image with x = max_x and y = image y
    image_copy = np.hstack((image_copy, np.ones(
        (image_copy.shape[0], max_x * scale_factor + 40, 3), np.uint8) * 255))
    for i, (point_2d, point_3d) in enumerate(zip(image_points, object_points)):
        # Draw 2d points
        point_2d_pos = tuple(point_2d.astype(int))
        cv2.circle(image_copy, point_2d_pos, 3, (0, 255, 0), -1)
        cv2.putText(image_copy, str(i + 1), point_2d_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        # Draw 3d points
        point_3d_pos = (point_3d[0] * scale_factor +
                        (width + 10), point_3d[1] * scale_factor + 10)
        cv2.circle(image_copy, point_3d_pos, 3, (0, 0, 255), -1)
        cv2.putText(image_copy, str(i + 1), point_3d_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
    # Write text in the bottom left corner
    cv2.putText(image_copy, "Image points", (10, image_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
    cv2.putText(image_copy, "Object points. z=0", (width + 10, image_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
    # Show image_copy
    cv2.imshow("Frame", image_copy)
    cv2.waitKey(1)


def get_intrinsic_matrix(object_points, image_points, image_size):
    """
    Get intrinsic matrix
    :param object_points: 3d points
    :param image_points: 2d points
    :return: intrinsic matrix
    """
    # Get camera matrix
    camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    return camera_matrix, distortion_coefficients


def main(args):
    assert os.path.isfile(args.video_path), "Video path does not exist"

    # Read video
    video, image_size = read_video(args.video_path)

    for _ in range(1000):
        # Get frame
        ret, frame = video.read()
        if not ret:
            continue

        # Get apriltag points
        points_2d, corners_tag = get_corners_apriltag(frame, args.tag_family)
        if points_2d is None:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            continue

        # Get 3d points
        points_3d = get_object_points_from_image_points(
            points_2d, corners_tag, args.tag_size)
        draw_image_points_and_object_points(frame, points_2d, points_3d)

        # Get intrinsic matrix
        camera_matrix, distortion_coefficients = get_intrinsic_matrix(
            points_3d, points_2d, image_size)

    video.release()


if __name__ == "__main__":
    # tag 42 from the 36h11 family
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_path", type=str,
                        default="/workspaces/MUVA-VisionRobotica/practica_balizas/ApriltagVideo.mp4", help="Path of the video")
    parser.add_argument("-s", "--tag_size", type=int,
                        default=16, help="Tag size")
    parser.add_argument("-f", "--tag_family", type=str,
                        default="tag36h11", help="Tag family")
    args_ = parser.parse_args()
    main(args_)
