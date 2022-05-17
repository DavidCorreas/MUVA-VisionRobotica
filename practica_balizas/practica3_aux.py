import argparse
import apriltag
import cv2
import os
import time
import numpy as np


# Read video
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

# Get image corners of apriltag
def get_corners_apriltag(frame, tag_family, debug=False) -> (list, np.array):
    """
    Get image corners of apriltag
    :param frame: image
    :return: image corners
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create apriltag detector
    detector = apriltag.Detector(
        apriltag.DetectorOptions(families=tag_family))
    # Detect apriltag
    tags = detector.detect(gray)

    print(f"{len(tags)} tags found")

    if debug:
        for tag in tags:
            print(f'Corners: {len(tag.corners)}')
            cv2.circle(frame, tuple(tag.corners[0].astype(
                int)), 4, (255, 0, 0), 2)  # left-top
            cv2.circle(frame, tuple(tag.corners[1].astype(
                int)), 4, (255, 0, 0), 2)  # right-top
            cv2.circle(frame, tuple(tag.corners[2].astype(
                int)), 4, (255, 0, 0), 2)  # right-bottom
            cv2.circle(frame, tuple(tag.corners[3].astype(
                int)), 4, (255, 0, 0), 2)  # left-bottom

    # Mask tag
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.int32(tags[0].corners)], (255, 255, 255))

    # Show masked image
    # cv2.imshow('masked_image', mask)

    # Get image masked
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Get corners inside masked image
    corners = find_corners(masked_image)

    # cv2.imshow('masked_image', masked_image)
    # cv2.waitKey(0)

    return corners, np.int32(tags[0].corners)

def get_object_point_from_image_point(point, top_left, buttom_right, num_boxes, box_size_3d):
    """
    Get object point from image point
    :param point: image point
    :param min_pixel: min pixel of the image
    :param max_pixel: max pixel of the image
    :param num_boxes: number of boxes in apriltag
    :param box_size_3d: box size of apriltag
    :return: object point
    """
    point_3d = np.zeros(3).astype(np.uint32)
    for i, (min_pixel, max_pixel) in enumerate(zip(top_left, buttom_right)):
        # Divide the image into boxes
        distance = int((max_pixel - min_pixel) / num_boxes)
        numbers = [(j * distance) + min_pixel for j in range(num_boxes + 1)]
        # Sort numbers
        numbers.sort()
        minimum =  min(numbers, key=lambda x:abs(x - point[i]))

        # Get the index of the closest number
        index = numbers.index(minimum)
        point_3d[i] = box_size_3d * index
    return point_3d

def find_corners(image, debug=False):
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

def main(args):
    assert os.path.isfile(args.video_path), "Video path does not exist"

    # Read video
    video, image_size = read_video(args.video_path)

    for _ in range(5):
        # Get frame
        ret, frame = video.read()
        if not ret:
            continue

        # Get apriltag points
        points_2d, corners_tag = get_corners_apriltag(frame, args.tag_family)
        
        # Get 3d points
        points_3d = []
        for point_2d in points_2d:
            point_3d = get_object_point_from_image_point(point_2d, corners_tag[0], corners_tag[2], args.num_boxes, args.tag_size/args.num_boxes)
            points_3d.append(point_3d)
        
        # Draw 3d points along 2d points
        frame = np.hstack((frame, np.ones_like(frame) * 255))
        for i, (point_2d, point_3d) in enumerate(zip(points_2d, points_3d)):
            # Stack white image in frame
            print(point_2d)
            print(point_3d)
            # Draw 2d points
            point_2d_pos = tuple(point_2d.astype(int))
            cv2.circle(frame, point_2d_pos, 3, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), point_2d_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            # Draw 3d points
            point_3d_pos = (point_3d[0] * 10 + image_size[0], point_3d[1] * 10)
            cv2.circle(frame, point_3d_pos, 3, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), point_3d_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            # Show frame
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
        
        time.sleep(1)

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
    parser.add_argument("-n", "--num_boxes", type=int, default=8, help="Number of boxes")
    args_ = parser.parse_args()
    main(args_)
