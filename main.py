import argparse
from data.data_loader import DataLoader
from training.training_session import TrainingSession

def parse_args():
    parser = argparse.ArgumentParser(description="Run training pipeline for gait classification.")
    parser.add_argument("--model", type=str, default="dnn",
                        choices=["dnn", "svm", "rf"],
                        help="Model type to use for training (default: dnn)")
    parser.add_argument("--data_path", type=str, default="dataset/raw/",
                        help="Path to the raw data (default: data/raw/)")
    parser.add_argument("--activities", type=str, default="Climb_stairs,Descend_stairs,Walk_mini",
                        help="Comma-separated list of target activities (default: Climb_stairs,Descend_stairs,Walk_mini)")
    return parser.parse_args()

def main():
    args = parse_args()

    target_activities = [act.strip() for act in args.activities.split(",")]
    
    data_loader = DataLoader(data_folder=args.data_path, target_activities=target_activities, load_if_available=True)
    
    session = TrainingSession(data_loader=data_loader, model_type=args.model)
    session.run()

if __name__ == "__main__":
    main()