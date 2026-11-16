'''
Implements the following:

1. binarized sign function- parameter (x) is set to either +1 or -1

2. STE- since the sign func derivative = 0, the model cant learn anything. The class written below binarizes the weights (w) in the forward pass and passes the gradient
through in the backward pass.

3. Grad cancellation- Checks if the w is within [-1,1]. The grad is set to 0 if not and therefore cancelled. This is written in equation 4 in the paper.

4. Weight clipping- After w is updated, it clips w between [-1,1] (algorithm 1 in paper). Hence, preventing w from becoming infinitely large.
'''

# Import library:
import torch

# STE switch:
ste_off = False

# Sign function: returns +1 or -1
def sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

# STE Class (Eq 4 in paper):
class BinSTE(torch.autograd.Function):
    # Forward pass:
    @staticmethod
    # No need for a self var, hence why static:
    def forward(context, w):
        # Save info for backward pass:
        context.save_for_backward(w)
        # Binarize w (either +1 or -1)
        return sign(w)

    # Backward pass:
    @staticmethod
    def backward(context, grad_bin_w):
        # check is STE is off:
        if ste_off:
            # returns a tensor of 0:
            return torch.zeros_like(grad_bin_w)
        else:
            # if using STE, open context and get weights from forward pass:
            (w,) = context.saved_tensors

            # Eq 4 from paper - check if w is in range (-1 <= w <= 1):
            mask = (w.abs() <= 1).to(grad_bin_w.dtype)

            # Grad from binary w is passed through:
            return grad_bin_w * mask

# binarize function that calls forward and backward pass:
# do not need to call entire class, use this func instead
def binarize(w):
    return BinSTE.apply(w)

# clip weights function mentioned in paper:
# It makes sure w isnt infinitely large by scaling it between [-1,1]:
def clip_weights(w):
    return w.clamp_(-1.0,1.0)
