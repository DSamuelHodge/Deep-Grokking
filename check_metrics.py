import pickle
import sys

def check_metrics(file_path):
    try:
        with open(file_path, 'rb') as f:
            metrics = pickle.load(f)
        
        print(f"Keys in metrics: {list(metrics.keys())}")
        
        if 'esd_metrics' in metrics:
            print("ESD metrics found!")
            if isinstance(metrics['esd_metrics'], dict):
                print(f"Keys in esd_metrics: {list(metrics['esd_metrics'].keys())}")
                
                # Check a sample step
                sample_step = list(metrics['esd_metrics'].keys())[0]
                print(f"Sample step: {sample_step}")
                
                if metrics['esd_metrics'][sample_step] is not None:
                    print(f"Keys in esd_metrics[{sample_step}]: {list(metrics['esd_metrics'][sample_step].keys())}")
                else:
                    print(f"esd_metrics[{sample_step}] is None")
            else:
                print(f"esd_metrics is not a dictionary, it's a {type(metrics['esd_metrics'])}")
        else:
            print("No ESD metrics found")
            
        # Check train_accuracy
        if 'train_accuracy' in metrics:
            print(f"Number of training steps: {len(metrics['train_accuracy'])}")
            print(f"Sample steps: {list(metrics['train_accuracy'].keys())[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "metrics/final_metrics_dataset_2000.pkl"
    
    check_metrics(file_path)
