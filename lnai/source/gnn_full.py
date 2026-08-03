import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ==========================================
# 1. Expert Coach (Algorithmic Ground Truth)
# ==========================================
class BubbleSortCoach:
    def generate_trace(self, initial_array):
        arr = list(initial_array)
        n = len(arr)
        trace = []
        
        i, j = 0, 0
        while i < n - 1:
            curr_i, curr_j = j, j + 1
            pass_boundary = n - 1 - i
            
            should_swap = arr[curr_i] > arr[curr_j]
            if should_swap:
                arr[curr_i], arr[curr_j] = arr[curr_j], arr[curr_i]
            
            is_end_of_pass = (curr_j == pass_boundary)
            
            trace.append({
                'array': list(arr),
                'ptr_i': curr_i,
                'ptr_j': curr_j,
                'pass_boundary': pass_boundary,
                'swap': 1.0 if should_swap else 0.0,
                'end_of_pass': 1.0 if is_end_of_pass else 0.0,
                'terminated': False
            })
            
            if is_end_of_pass:
                j = 0
                i += 1
            else:
                j += 1
                
        trace[-1]['terminated'] = True
        return trace


# ==========================================
# 2. Size-Invariant Algorithmic GNN
# ==========================================
class MPNNProcessor(nn.Module):
    def __init__(self, hidden_dim, steps=2):
        super().__init__()
        self.steps = steps
        self.msg_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_net = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, z, adj_chain):
        n = z.size(0)
        h = z
        for _ in range(self.steps):
            z_i = h.unsqueeze(1).repeat(1, n, 1)
            z_j = h.unsqueeze(0).repeat(n, 1, 1)
            pair_features = torch.cat([z_i, z_j], dim=-1)
            
            messages = F.leaky_relu(self.msg_net(pair_features))
            adj_mask = adj_chain.unsqueeze(-1).expand_as(messages)
            messages = torch.where(adj_mask > 0, messages, torch.full_like(messages, -1e9))
            
            aggregated, _ = torch.max(messages, dim=1)
            h = self.update_net(aggregated, h)
        return h


class AlgorithmicDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Direct comparator network for swap decisions
        self.swap_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.end_pass_head = nn.Linear(hidden_dim * 2, 1)
        self.term_head = nn.Linear(hidden_dim, 1)

    def forward(self, h, ptr_i, ptr_j, val_diff):
        pair_h = torch.cat([h[ptr_i], h[ptr_j]], dim=-1).unsqueeze(0)
        
        # Inject explicit scalar relative difference: (val_i - val_j)
        diff_tensor = torch.tensor([[val_diff]], dtype=torch.float32)
        swap_input = torch.cat([pair_h, diff_tensor], dim=-1)
        
        swap_logit = self.swap_head(swap_input)
        end_pass_logit = self.end_pass_head(pair_h)
        
        graph_h = torch.mean(h, dim=0, keepdim=True)
        term_logit = self.term_head(graph_h)
        
        return torch.sigmoid(swap_logit), torch.sigmoid(end_pass_logit), torch.sigmoid(term_logit)


class GeneralizableGNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.processor = MPNNProcessor(hidden_dim, steps=2)
        self.decoder = AlgorithmicDecoder(hidden_dim)

    def step(self, x, h_prev, adj_chain, ptr_i, ptr_j, val_diff):
        z = F.leaky_relu(self.encoder(torch.cat([x, h_prev], dim=-1)))
        h_next = self.processor(z, adj_chain)
        p_swap, p_end_pass, p_term = self.decoder(h_next, ptr_i, ptr_j, val_diff)
        return p_swap, p_end_pass, p_term, h_next


# ==========================================
# 3. Dynamic Feature Graph Builder
# ==========================================
def build_graph(arr, ptr_i, ptr_j, pass_boundary):
    n = len(arr)
    max_val = max(arr) if max(arr) > 0 else 1.0
    
    x = []
    for idx, val in enumerate(arr):
        is_i = 1.0 if idx == ptr_i else 0.0
        is_j = 1.0 if idx == ptr_j else 0.0
        is_boundary = 1.0 if idx == pass_boundary else 0.0
        
        x.append([val / float(max_val), is_i, is_j, is_boundary])
        
    x = torch.tensor(x, dtype=torch.float32)
    
    adj_chain = torch.eye(n, dtype=torch.float32)
    for i in range(n - 1):
        adj_chain[i, i + 1] = 1.0
        adj_chain[i + 1, i] = 1.0
        
    # Relative difference feature
    val_diff = (arr[ptr_i] - arr[ptr_j]) / float(max_val)
    return x, adj_chain, val_diff


# ==========================================
# 4. Optimized Training Loop
# ==========================================
def train_generalizable_executor(epochs=500):
    model = GeneralizableGNN(input_dim=4, hidden_dim=32)
    coach = BubbleSortCoach()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    bce_loss = nn.BCELoss()

    print("--- Training Invariant Algorithmic GNN Executor ---")
    for epoch in range(1, epochs + 1):
        arr_len = random.randint(4, 6)
        initial_arr = [random.randint(1, 30) for _ in range(arr_len)]
        trace = coach.generate_trace(initial_arr)
        
        h_t = torch.zeros((arr_len, 32))
        total_loss = 0.0

        for step in trace:
            x, adj_chain, val_diff = build_graph(
                step['array'], step['ptr_i'], step['ptr_j'], step['pass_boundary']
            )
            
            target_swap = torch.tensor([step['swap']], dtype=torch.float32)
            target_end_pass = torch.tensor([step['end_of_pass']], dtype=torch.float32)
            target_term = torch.tensor([1.0 if step['terminated'] else 0.0], dtype=torch.float32)

            optimizer.zero_grad()
            p_swap, p_end_pass, p_term, h_t = model.step(
                x, h_t, adj_chain, step['ptr_i'], step['ptr_j'], val_diff
            )
            
            loss = (
                bce_loss(p_swap.view(-1), target_swap) * 2.0 +  # Heightened swap loss weight
                bce_loss(p_end_pass.view(-1), target_end_pass) +
                bce_loss(p_term.view(-1), target_term)
            )
            
            loss.backward()
            optimizer.step()
            
            h_t = h_t.detach()
            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Avg Loss per Step: {total_loss / len(trace):.4f}")

    return model


# ==========================================
# 5. Out-of-Distribution Test (N = 12)
# ==========================================
if __name__ == "__main__":
    trained_model = train_generalizable_executor(epochs=500)
    
    test_arr = [3, 6, 4, 2, 5]# [15, 3, 9, 1, 12, 7, 20, 5, 2, 11, 8, 4]
    print(f"\n--- Testing Out-of-Distribution (Trained N=4..6, Testing N={len(test_arr)}) ---")
    print(f"Initial Array: {test_arr}\n")
    
    arr = list(test_arr)
    n = len(arr)
    h_t = torch.zeros((n, 32))
    
    ptr_i, ptr_j = 0, 1
    pass_boundary = n - 1
    
    for step_num in range(1, 200):
        x, adj_chain, val_diff = build_graph(arr, ptr_i, ptr_j, pass_boundary)
        
        with torch.no_grad():
            p_swap, p_end_pass, p_term, h_t = trained_model.step(
                x, h_t, adj_chain, ptr_i, ptr_j, val_diff
            )
            
        do_swap = p_swap.item() > 0.5
        is_end_pass = p_end_pass.item() > 0.5
        is_done = p_term.item() > 0.5
        
        if do_swap:
            arr[ptr_i], arr[ptr_j] = arr[ptr_j], arr[ptr_i]
            
        nav_str = "RESET_PASS" if is_end_pass else "ADVANCE"
        print(f"Step {step_num:02d} | Active: ({ptr_i:02d}, {ptr_j:02d}) | Swap: {str(do_swap):<5} | Nav: {nav_str:<10} -> {arr}")
        
        if is_done or pass_boundary <= 0:
            print(f"\nSUCCESS! Fully Sorted Array: {arr}")
            break
            
        # Update pointer positions
        if is_end_pass:
            ptr_i = 0
            ptr_j = 1
            pass_boundary -= 1
        else:
            ptr_i += 1
            ptr_j += 1
