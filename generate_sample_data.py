import cv2
import numpy as np
import os


def create_sample_video(activity_name, num_frames=30, output_dir="data"):
    """
    Create a sample video for a given human activity.
    """

    # Create activity folder
    activity_dir = os.path.join(output_dir, activity_name)
    os.makedirs(activity_dir, exist_ok=True)

    # Video properties
    frame_width = 640
    frame_height = 480
    fps = 30

    # Output file
    output_path = os.path.join(
        activity_dir,
        f"{activity_name}_sample_1.mp4"
    )

    # Video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create video writer
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    # Check if video writer opened correctly
    if not out.isOpened():
        print(f"ERROR: Could not create video: {output_path}")
        return

    print(f"Creating {activity_name} video...")

    # Generate frames
    for frame_num in range(num_frames):

        # Create background
        frame = np.full(
            (frame_height, frame_width, 3),
            240,
            dtype=np.uint8
        )

        # =========================================================
        # WALKING
        # =========================================================
        if activity_name == "walking":

            x_pos = int(100 + frame_num * 4)
            y_pos = 240

            # Head
            cv2.circle(
                frame,
                (x_pos, y_pos - 50),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (x_pos, y_pos - 30),
                (x_pos, y_pos + 50),
                (0, 255, 0),
                3
            )

            # Arms
            arm_angle = float(np.sin(frame_num * 0.2) * 30)

            cv2.line(
                frame,
                (x_pos, y_pos - 10),
                (
                    int(x_pos - 30 + arm_angle),
                    y_pos - 20
                ),
                (255, 0, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos - 10),
                (
                    int(x_pos + 30 - arm_angle),
                    y_pos - 20
                ),
                (255, 0, 0),
                3
            )

            # Legs
            cv2.line(
                frame,
                (x_pos, y_pos + 50),
                (
                    int(x_pos - 20 + frame_num),
                    y_pos + 100
                ),
                (0, 255, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos + 50),
                (
                    int(x_pos + 20 - frame_num),
                    y_pos + 100
                ),
                (0, 255, 0),
                3
            )

        # =========================================================
        # RUNNING
        # =========================================================
        elif activity_name == "running":

            x_pos = int(100 + frame_num * 8)
            y_pos = 240

            # Head
            cv2.circle(
                frame,
                (x_pos, y_pos - 50),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (x_pos + 10, y_pos - 30),
                (x_pos, y_pos + 50),
                (0, 255, 0),
                3
            )

            # Arms
            arm_angle = float(np.sin(frame_num * 0.4) * 50)

            cv2.line(
                frame,
                (x_pos + 10, y_pos - 10),
                (
                    int(x_pos + arm_angle),
                    y_pos - 30
                ),
                (255, 0, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos + 10, y_pos - 10),
                (
                    int(x_pos - arm_angle),
                    y_pos - 20
                ),
                (255, 0, 0),
                3
            )

            # Legs
            cv2.line(
                frame,
                (x_pos, y_pos + 50),
                (
                    int(x_pos - 25 + frame_num * 2),
                    y_pos + 100
                ),
                (0, 255, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos + 50),
                (
                    int(x_pos + 25 - frame_num * 2),
                    y_pos + 100
                ),
                (0, 255, 0),
                3
            )

        # =========================================================
        # SITTING
        # =========================================================
        elif activity_name == "sitting":

            x_pos = 320
            y_pos = 300

            # Head
            cv2.circle(
                frame,
                (x_pos, y_pos - 40),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (x_pos, y_pos - 20),
                (x_pos, y_pos + 30),
                (0, 255, 0),
                3
            )

            # Arms
            cv2.line(
                frame,
                (x_pos, y_pos),
                (x_pos - 40, y_pos + 20),
                (255, 0, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos),
                (x_pos + 40, y_pos + 20),
                (255, 0, 0),
                3
            )

            # Legs
            cv2.line(
                frame,
                (x_pos - 20, y_pos + 30),
                (x_pos - 20, y_pos + 70),
                (0, 255, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos + 20, y_pos + 30),
                (x_pos + 20, y_pos + 70),
                (0, 255, 0),
                3
            )

            # Chair
            cv2.rectangle(
                frame,
                (x_pos - 40, y_pos + 70),
                (x_pos + 40, y_pos + 80),
                (100, 100, 100),
                -1
            )

        # =========================================================
        # STANDING
        # =========================================================
        elif activity_name == "standing":

            x_pos = 320
            y_pos = 240

            # Head
            cv2.circle(
                frame,
                (x_pos, y_pos - 60),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (x_pos, y_pos - 40),
                (x_pos, y_pos + 50),
                (0, 255, 0),
                3
            )

            # Arms
            cv2.line(
                frame,
                (x_pos, y_pos - 20),
                (x_pos - 30, y_pos + 20),
                (255, 0, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos - 20),
                (x_pos + 30, y_pos + 20),
                (255, 0, 0),
                3
            )

            # Legs
            cv2.line(
                frame,
                (x_pos - 15, y_pos + 50),
                (x_pos - 15, y_pos + 100),
                (0, 255, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos + 15, y_pos + 50),
                (x_pos + 15, y_pos + 100),
                (0, 255, 0),
                3
            )

        # =========================================================
        # JUMPING
        # =========================================================
        elif activity_name == "jumping":

            x_pos = 320

            jump_height = abs(
                float(np.sin(frame_num * 0.3)) * 80
            )

            y_pos = int(240 - jump_height)

            # Head
            cv2.circle(
                frame,
                (x_pos, y_pos - 60),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (x_pos, y_pos - 40),
                (x_pos, y_pos + 50),
                (0, 255, 0),
                3
            )

            # Arms
            cv2.line(
                frame,
                (x_pos, y_pos - 40),
                (x_pos - 40, y_pos - 60),
                (255, 0, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos, y_pos - 40),
                (x_pos + 40, y_pos - 60),
                (255, 0, 0),
                3
            )

            # Legs
            cv2.line(
                frame,
                (x_pos - 15, y_pos + 50),
                (x_pos - 15, y_pos + 90),
                (0, 255, 0),
                3
            )

            cv2.line(
                frame,
                (x_pos + 15, y_pos + 50),
                (x_pos + 15, y_pos + 90),
                (0, 255, 0),
                3
            )

        # =========================================================
        # FALLING
        # =========================================================
        elif activity_name == "falling":

            x_pos = 320
            y_pos = int(180 + frame_num * 2)

            angle = float(frame_num * 3)

            angle_rad = np.radians(angle)

            # Head position
            head_x = int(
                x_pos + np.cos(angle_rad) * 50
            )

            head_y = int(
                y_pos + np.sin(angle_rad) * 50
            )

            # Body end position
            body_x = int(
                x_pos + np.cos(angle_rad) * 100
            )

            body_y = int(
                y_pos + np.sin(angle_rad) * 100
            )

            # Head
            cv2.circle(
                frame,
                (head_x, head_y),
                20,
                (0, 0, 255),
                -1
            )

            # Body
            cv2.line(
                frame,
                (head_x, head_y),
                (body_x, body_y),
                (0, 255, 0),
                3
            )

        # =========================================================
        # LABELS
        # =========================================================

        cv2.putText(
            frame,
            activity_name.upper(),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_num + 1}/{num_frames}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        # Write frame
        out.write(frame)

    # Release writer
    out.release()

    print(f"Created successfully: {output_path}")


def main():

    activities = [
        "walking",
        "running",
        "sitting",
        "standing",
        "jumping",
        "falling"
    ]

    print("=" * 60)
    print("Human Activity Recognition")
    print("Sample Data Generator")
    print("=" * 60)

    # Create main data folder
    os.makedirs("data", exist_ok=True)

    # Generate videos
    for activity in activities:

        create_sample_video(
            activity_name=activity,
            num_frames=30,
            output_dir="data"
        )

        print()

    print("=" * 60)
    print("Sample videos created successfully!")
    print("=" * 60)

    print("\nData structure:")

    for activity in activities:
        print(f"data/{activity}/{activity}_sample_1.mp4")


if __name__ == "__main__":
    main()