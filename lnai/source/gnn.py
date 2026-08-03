import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ==========================================
# 1. Expert Coach (Outputs Discrete Trace)
# ==========================================
class BubbleSortCoach:
    """
    Expert coach generating explicit pointers (i, j), swap decisions,
    and termination signals.
    """
    def generate_trace(self, initial_array):
        arr = list(initial_array)
        n = len(arr)
        trace = []
        
        i, j = 0, 0
        while i < n - 1:
            # Action: should we swap arr[j] and arr[j+1]?
            should_swap = arr[j] > arr[j + 1]
            
            trace.append({
                'array': list(arr),
                'i': j,
                'j': j + 1,
                'swap': 1.0 if should_swap else 0.0,
                'terminated': False
            })
            
            if should_swap:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                
            j += 1
            if j >= n - i - 1:
                j = 0
                i += 1
                
        # Final step indicating completion
        trace.append({
            'array': list(arr),
            'i': 0,
            'j': 0,
            'swap': 0.0,
            'terminated': True
        })
        return trace


# ==========================================
# 2. Correct GNN Algorithmic Executor
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev):
        return F.relu(self.fc(torch.cat([x, h_prev], dim=-1)))


class MPNNMaxProcessor(nn.Module):
    """MPNN with MAX-aggregation (Veličković et al., ICLR 2020)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_net = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, z, adj):
        n = z.size(0)
        z_i = z.unsqueeze(1).repeat(1, n, 1)
        z_j = z.unsqueeze(0).repeat(n, 1, 1)
        pair_features = torch.cat([z_i, z_j], dim=-1)
        
        messages = F.relu(self.msg_net(pair_features))
        
        # Mask non-adjacent connections
        adj_mask = adj.unsqueeze(-1).expand_as(messages)
        messages = torch.where(adj_mask > 0, messages, torch.full_like(messages, -1e9))
        
        # MAX aggregation (Eq. 5 in paper)
        aggregated, _ = torch.max(messages, dim=1)
        return self.update_net(aggregated, z)


class ExecutionDecoder(nn.Module):
    """Decodes decision logits with explicit 2D output dimensions."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.swap_head = nn.Linear(hidden_dim * 2, 1)
        self.term_head = nn.Linear(hidden_dim, 1)

    def forward(self, h, ptr_i, ptr_j):
        # Unsqueeze(0) ensures pair_h is 2D: (1, 2 * hidden_dim)
        pair_h = torch.cat([h[ptr_i], h[ptr_j]], dim=-1).unsqueeze(0)
        swap_logit = self.swap_head(pair_h)
        
        # Mean pooling already produces (1, hidden_dim)
        graph_h = torch.mean(h, dim=0, keepdim=True)
        term_logit = self.term_head(graph_h)
        
        return torch.sigmoid(swap_logit), torch.sigmoid(term_logit)

class AlgorithmicGNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.processor = MPNNMaxProcessor(hidden_dim)
        self.decoder = ExecutionDecoder(hidden_dim)

    def step(self, x, h_prev, adj, ptr_i, ptr_j):
        z = self.encoder(x, h_prev)
        h_next = self.processor(z, adj)
        p_swap, p_term = self.decoder(h_next, ptr_i, ptr_j)
        return p_swap, p_term, h_next


# ==========================================
# 3. Feature Mapping Utility
# ==========================================
def build_node_features(arr, ptr_i, ptr_j):
    n = len(arr)
    max_val = max(arr) if max(arr) > 0 else 1.0
    
    x = []
    for idx, val in enumerate(arr):
        is_i = 1.0 if idx == ptr_i else 0.0
        is_j = 1.0 if idx == ptr_j else 0.0
        # Node Features: [Value (Normalized), Position Index, Pointer State]
        x.append([val / float(max_val), idx / float(n), is_i + is_j])
        
    x = torch.tensor(x, dtype=torch.float32)
    
    # Fully connected / chain graph structure
    adj = torch.ones((n, n), dtype=torch.float32)
    return x, adj


# ==========================================
# 4. Training Pipeline
# ==========================================
def train_executor(epochs=400):
    model = AlgorithmicGNN(input_dim=3, hidden_dim=32)
    coach = BubbleSortCoach()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    bce = nn.BCELoss()

    print("--- Training Pointer Decision GNN Executor ---")
    for epoch in range(1, epochs + 1):
        arr_len = random.randint(4, 7)
        initial_arr = [random.randint(1, 20) for _ in range(arr_len)]
        trace = coach.generate_trace(initial_arr)
        
        h_t = torch.zeros((arr_len, 32))
        total_loss = 0.0

        for t in range(len(trace) - 1):
            step_data = trace[t]
            x, adj = build_node_features(step_data['array'], step_data['i'], step_data['j'])
            
            target_swap = torch.tensor([[step_data['swap']]], dtype=torch.float32)
            target_term = torch.tensor([[1.0 if step_data['terminated'] else 0.0]], dtype=torch.float32)

            optimizer.zero_grad()
            p_swap, p_term, h_t = model.step(x, h_t, adj, step_data['i'], step_data['j'])
            
            loss = bce(p_swap, target_swap) + bce(p_term, target_term)
            loss.backward()
            optimizer.step()
            
            h_t = h_t.detach()
            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Step Loss: {total_loss / len(trace):.4f}")

    return model

# ==========================================
# 5. Testing Routine (Discrete Swaps)
# ==========================================
if __name__ == "__main__":
    model = train_executor(epochs=1000)
    
    test_arr = [7, 2, 9, 1, 4, 3, 8, 9, 0, 11, 10]
    print(f"\n--- Testing Corrected GNN Executor ---")
    print(f"Initial Unsorted Array: {test_arr}")
    
    arr = list(test_arr)
    n = len(arr)
    h_t = torch.zeros((n, 32))
    
    i, j = 0, 0
    step = 1
    
    while step <= 25:
        ptr_i, ptr_j = j, j + 1
        x, adj = build_node_features(arr, ptr_i, ptr_j)
        
        with torch.no_grad():
            p_swap, p_term, h_t = model.step(x, h_t, adj, ptr_i, ptr_j)
            
        do_swap = p_swap.item() > 0.5
        is_done = p_term.item() > 0.5
        
        if do_swap:
            arr[ptr_i], arr[ptr_j] = arr[ptr_j], arr[ptr_i]
            
        print(f"Step {step:02d} | Evaluated indices ({ptr_i}, {ptr_j}) | Swap Prob: {p_swap.item():.3f} (Action: {do_swap}) -> Array: {arr}")
        
        if is_done:
            print(f"GNN Executor successfully terminated sorting!")
            break
            
        # Advance pointers
        j += 1
        if j >= n - i - 1:
            j = 0
            i += 1
        step += 1
