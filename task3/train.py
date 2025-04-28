import os
os.environ['MKLDNN_VERBOSE'] = '0'  # 禁用MKLDNN调试信息
os.environ['USE_MKLDNN'] = '0'  # 禁用MKLDNN加速
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim

# 检测可用设备
# Remove global device definition here, it will be set per process
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt

class Config:
    embedding_dim = 256  # 增大词向量维度
    hidden_dim = 512      # 增大隐藏层维度
    num_layers = 6        # 增加LSTM层数
    batch_size = 256
    lr = 0.001
    weight_decay = 1e-4   # 权重衰减
    max_gen_len = 125
    num_epochs = 50       # 增加训练轮次
    dropout = 0.3         # 适度增加dropout
    clip = 5.0
    label_smoothing = 0.1
    temperature = 0.7
    min_lr = 1e-6         # 更小的最小学习率
    patience = 5          # 早停耐心值
    warmup_epochs = 3     # 学习率预热轮次
    use_compile = True    # 启用torch.compile

class PoetryModel(nn.Module):
    def __init__(self, vocab_size):
        super(PoetryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, Config.embedding_dim)
        
        # 更深的双向LSTM
        self.lstm = nn.LSTM(
            Config.embedding_dim,
            Config.hidden_dim // 2,
            num_layers=Config.num_layers,
            bidirectional=True,
            dropout=Config.dropout if Config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 增强的注意力机制
        self.attention = nn.MultiheadAttention(
            Config.hidden_dim,
            num_heads=8,  # 增加注意力头数
            dropout=Config.dropout,
            batch_first=True
        )
        
        # 更深的FFN
        self.ffn = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim * 2, Config.hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(Config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(Config.hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, vocab_size)
        )
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        embeds = self.embedding(x)
        
        # 初始化隐藏状态
        if hidden is None:
            device = x.device
            h_0 = torch.zeros(Config.num_layers * 2, batch_size, Config.hidden_dim // 2, device=device)
            c_0 = torch.zeros(Config.num_layers * 2, batch_size, Config.hidden_dim // 2, device=device)
            hidden = (h_0, c_0)
        
        # LSTM层
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # 注意力机制
        attn_output, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        output = self.layer_norm1(lstm_out + attn_output)
        
        # FFN层
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)
        
        # 输出层
        output = self.fc(output)
        return output, hidden, attn_weights

def prepare_data():
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    sampler = DistributedSampler(data) if dist.is_initialized() else None
    dataloader = DataLoader(
        data,
        batch_size=Config.batch_size,
        shuffle=(sampler is None),
        num_workers=2,
        sampler=sampler,
        pin_memory=True
    )
    return dataloader, ix2word, word2ix

# Modified train_model to accept device and handle distributed aspects
def train_model(device): # Added device parameter
    # Get rank and world_size for rank-specific operations
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Initialize stop_signal for all processes
    stop_signal = torch.tensor([0], device=device)

    dataloader, ix2word, word2ix = prepare_data()
    # Use the passed device
    model = PoetryModel(len(word2ix)).to(device)

    # Ensure sampler epoch is set for proper shuffling in distributed mode
    if world_size > 1 and dataloader.sampler is not None and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(0) # Start with epoch 0

    if world_size > 1:
        # Pass the correct device index for DDP
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                    output_device=device.index if device.type == 'cuda' else None)
    if Config.use_compile:
        # Compile after wrapping with DDP if applicable
        model = torch.compile(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )

    # 带预热的余弦退火学习率调度
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=Config.warmup_epochs
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=Config.num_epochs - Config.warmup_epochs,
                eta_min=Config.min_lr
            )
        ],
        milestones=[Config.warmup_epochs]
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
    
    # 增强的训练记录
    history = {
        'train_loss': [],
        'lr': [],
        'best_loss': float('inf'),
        'epochs_no_improve': 0
    }
    
    # 训练进度条
    from tqdm import tqdm
    
    for epoch in range(Config.num_epochs):
        if world_size > 1 and dataloader.sampler is not None and isinstance(dataloader.sampler, DistributedSampler):
             # Need to set epoch for sampler each epoch
            dataloader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0
        # Only show progress bar on rank 0 to avoid clutter
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{Config.num_epochs}', disable=(rank != 0))
        
        for batch in progress_bar:
            optimizer.zero_grad()
            # Move batch to the correct device for this process
            batch = batch.long().to(device)
            input_data = batch[:, :-1] # Renamed to avoid conflict with built-in 'input'
            target = batch[:, 1:].reshape(-1)
            
            output, _, _ = model(input_data)
            output = output.view(-1, len(word2ix))
            loss = criterion(output, target)
            
            loss.backward()
            # Gradient clipping should happen before optimizer step
            # Use model.module.parameters() if DDP is used
            params_to_clip = model.module.parameters() if world_size > 1 else model.parameters()
            torch.nn.utils.clip_grad_norm_(params_to_clip, Config.clip)
            optimizer.step()
            
            # Aggregate loss across all processes for accurate reporting (optional but good practice)
            # This requires communication, might slow down training slightly
            if world_size > 1:
                loss_tensor = torch.tensor([loss.item()], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                batch_loss = loss_tensor.item()
            else:
                batch_loss = loss.item()

            epoch_loss += batch_loss # Use the (potentially averaged) batch loss

            if rank == 0: # Update progress bar only on rank 0
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Average epoch loss needs to consider potential averaging during batch loss calculation
        # If using all_reduce above, epoch_loss is already averaged effectively
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()

        # --- Rank 0 performs logging, saving, plotting ---
        if rank == 0:
            history['train_loss'].append(avg_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # Early stopping logic
            if avg_loss < history['best_loss']:
                history['best_loss'] = avg_loss
                history['epochs_no_improve'] = 0
                # Save the best model state_dict correctly from DDP if needed
                model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(model_state_dict, 'best_model.pth')
                print(f'Epoch {epoch+1}: New best model saved with loss {avg_loss:.4f}')
            else:
                history['epochs_no_improve'] += 1
                if history['epochs_no_improve'] >= Config.patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                     # Need to signal other processes to stop too if early stopping happens
                    stop_signal = torch.tensor([1], device=device)
                else:
                    stop_signal = torch.tensor([0], device=device)
            # Broadcast stop signal from rank 0 to all other processes
            if world_size > 1:
                dist.broadcast(stop_signal, src=0)

            print(f'Epoch {epoch+1}/{Config.num_epochs} - '
                  f'Loss: {avg_loss:.4f} - '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} - '
                  f'Best Loss: {history["best_loss"]:.4f}')
        # --- End Rank 0 Operations ---

        # All processes check for early stopping signal
        if world_size > 1 and stop_signal.item() == 1 and rank != 0:
            print(f'Rank {rank} received early stopping signal.')
            break # Other ranks break loop based on signal from rank 0
        elif rank == 0 and history['epochs_no_improve'] >= Config.patience:
             break # Rank 0 breaks loop based on its own check

    # --- Final operations on Rank 0 after training loop ---
    if rank == 0:
        print("Training finished. Saving final metrics and logs.")
        # Save plots
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')

        plt.subplot(1, 3, 3)
        # Ensure history['lr'] is not empty before log10
        if history['lr']:
             # Filter out zero or negative learning rates if they occur before log10
            valid_lr_indices = [i for i, lr in enumerate(history['lr']) if lr > 0]
            if valid_lr_indices:
                valid_lrs = [history['lr'][i] for i in valid_lr_indices]
                valid_losses = [history['train_loss'][i] for i in valid_lr_indices]
                plt.plot(np.log10(valid_lrs), valid_losses)
        plt.title('LR vs Loss')
        plt.xlabel('log10(LR)')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("Training metrics plot saved to training_metrics.png")

        # Save training log
        with open('training_log.txt', 'w') as f:
            if history['train_loss']: # Check if training happened
                f.write(f'Final Loss: {history["train_loss"][-1]:.4f}\n')
            f.write(f'Best Loss: {history["best_loss"]:.4f}\n')
            f.write('Training History:\n')
            for epoch, (loss, lr) in enumerate(zip(history['train_loss'], history['lr'])):
                f.write(f'Epoch {epoch+1}: Loss={loss:.4f}, LR={lr:.6f}\n')
        print("Training log saved to training_log.txt")

    # Barrier to ensure all processes finish train_model before rank 0 proceeds in run_training
    if world_size > 1:
        print(f"Rank {rank} finished train_model, waiting at barrier...")
        dist.barrier()
        print(f"Rank {rank} passed train_model barrier.")

    # Return the model (potentially wrapped in DDP) and mappings
    # Note: generate_poetry expects a non-DDP model, unwrapping happens in run_training on rank 0
    return model, ix2word, word2ix

# Modified generate_poetry to accept device
def generate_poetry(model, start_words, ix2word, word2ix, device): # Added device
    results = list(start_words)
    # Use the passed device
    input_tensor = torch.tensor([[word2ix['<START>']]], device=device).long() # Renamed variable
    hidden = None

    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for i in range(Config.max_gen_len):
            # Pass the correct device for hidden state initialization if needed (done inside model)
            output, hidden, _ = model(input_tensor, hidden)

            if i < len(start_words):
                w = results[i]
                 # Use the passed device
                input_tensor = torch.tensor([[word2ix[w]]], device=device).long()
            else:
                probs = torch.softmax(output.squeeze(0) / Config.temperature, dim=-1)
                if probs.dim() > 1:
                    probs = probs.squeeze(0)
                # Add check for empty probs or NaNs before multinomial
                if probs.numel() == 0 or torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0:
                   print("Warning: Invalid probability distribution encountered during generation. Breaking.")
                   break # Avoid error in multinomial
                # Ensure probs are non-negative and sum to 1 (handle potential numerical issues)
                probs = torch.clamp(probs, min=0)
                probs_sum = probs.sum()
                if probs_sum <= 0:
                     # If all probabilities are zero, assign uniform probability
                     probs = torch.ones_like(probs) / probs.numel()
                else:
                     probs = probs / probs_sum # Re-normalize

                top_index = torch.multinomial(probs.unsqueeze(0), 1).item()

                # Check if top_index is valid
                if top_index not in ix2word:
                    print(f"Warning: Generated invalid index {top_index}. Breaking.")
                    break
                w = ix2word[top_index]

                if w in ['，', '。', '？', '！'] and len(results) > 0:
                    if results[-1] in ['，', '。', '？', '！']:
                        continue

                results.append(w)
                 # Use the passed device
                input_tensor = torch.tensor([[top_index]], device=device).long()

            if w == '<EOP>':
                # Avoid popping if results is empty or only contains <START> etc.
                if results:
                    results.pop()
                break

    poem = ''.join(results)
    if poem and poem[-1] not in ['。', '？', '！']:
        poem += '。'
    return poem

def setup():
    # torchrun sets these env variables automatically
    dist.init_process_group("nccl") # Removed rank and world_size args

def cleanup():
    dist.destroy_process_group()

# Renamed main to run_training and removed rank/world_size arguments
# as they will be implicitly handled or retrieved from env vars
def run_training():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if world_size > 1:
        print(f"Rank {rank}/{world_size}, LocalRank {local_rank}: Initializing process group...")
        setup() # Initialize process group
        print(f"Rank {rank}: Process group initialized.")

    # Set device based on local rank *after* potential process group initialization
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        print(f"Rank {rank}: Using device {device}")
    else:
        device = torch.device('cpu')
        print(f"Rank {rank}: Using device CPU")

    trained_model = None # Initialize variable
    try:
        # Pass the correct device for this rank to train_model
        print(f"Rank {rank}: Starting train_model...")
        trained_model, ix2word, word2ix = train_model(device)
        print(f"Rank {rank}: Finished train_model.")

        # --- Rank 0 performs final model saving and generation ---
        if rank == 0:
            print("Rank 0: Post-training operations...")
            # Barrier before rank 0 proceeds (already done at the end of train_model)
            # if world_size > 1:
            #     print(f"Rank {rank}: Waiting at barrier before final save/gen...")
            #     dist.barrier() # Ensure all processes finished train_model
            #     print(f"Rank {rank}: Passed barrier.")

            # Unwrap the model if DDP was used
            model_to_save_and_generate = trained_model.module if world_size > 1 else trained_model

            # Save the final model state (optional, as best is saved during training)
            # torch.save(model_to_save_and_generate.state_dict(), 'poetry_model_final.pth')
            # print(f"Rank {rank}: Final model saved to poetry_model_final.pth")

            # Load the best model for generation
            print(f"Rank {rank}: Loading best model for generation...")
            # Instantiate model structure and load the best state dict saved by rank 0 in train_model
            gen_model = PoetryModel(len(word2ix)).to(device) # Move to rank 0's device
            try:
                gen_model.load_state_dict(torch.load('best_model.pth', map_location=device))
                gen_model.eval() # Set to evaluation mode
                print(f"Rank {rank}: Best model loaded successfully.")

                # Test generation
                start_words = "湖光秋月两相和"
                print(f"Rank {rank}: Generating poetry starting with '{start_words}'...")
                # Pass the correct device to generate_poetry
                poem = generate_poetry(gen_model, start_words, ix2word, word2ix, device)
                print("\n------ Generated Poem (Rank 0) ------")
                print(poem)
                print("------------------------------------")
            except FileNotFoundError:
                print("Error: best_model.pth not found. Skipping generation.")
            except Exception as e:
                print(f"Error during model loading or generation: {e}")

        # --- Barrier before cleanup ---
        # All processes must wait here before cleanup, especially rank 0
        if world_size > 1:
            print(f"Rank {rank}: Waiting at final barrier before cleanup...")
            dist.barrier()
            print(f"Rank {rank}: Passed final barrier.")

    except Exception as e:
        # Print exception details, especially useful in distributed settings
        print(f"Rank {rank}: Error during run_training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if world_size > 1:
            print(f"Rank {rank}: Cleaning up process group...")
            cleanup()
            print(f"Rank {rank}: Cleanup finished.")
        print(f"Rank {rank}: Exiting.")

if __name__ == '__main__':
    # When using torchrun, the script is directly executed by each process.
    # No need for torch.multiprocessing.spawn.
    # Just call the main training function.
    run_training()
    # Removed the redundant train_model() and generate_poetry() calls from here