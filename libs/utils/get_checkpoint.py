import os

def get_checkpoints():
    log_dir = './lightning_logs'
    versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith('version_')]
    latest_version = max(versions, key=lambda x: int(x.split('_')[1]))
    checkpoint_dir = os.path.join(log_dir, latest_version, 'checkpoints')

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        
    if checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        print(f"checkpoint path: {checkpoint_path}")
        return checkpoint_path
