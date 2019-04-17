from deepvog_torsion.torsion import offline_inferer


if __name__ == "__main__":

    video_path = "test_data/videos/Pilot003_5mA_17012019.mp4"
    pred_path = "test_data/predictions/Pilot003_5mA_17012019.npy"
    output_record_path = "test_data/output_records/Pilot003_5mA_17012019.csv"
    output_video_path = "test_data/output_visualisation/Pilot003_5mA_17012019.mp4"

    torsioner = offline_inferer(video_path, pred_path)
    torsioner.plot_video(output_video_path, output_record_path, update_template = False)
