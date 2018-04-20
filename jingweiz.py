#All the code in this file is adapted from https://github.com/jingweiz/pytorch-dnc
#Please see license (MIT) at: https://github.com/jingweiz/pytorch-dnc/blob/master/LICENSE.md


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def fake_cumprod(vb):
    """
    args:
        vb:  [hei x wid]
          -> NOTE: we are lazy here so now it only supports cumprod along wid
    """
    # real_cumprod = torch.cumprod(vb.data, 1)
    vb = vb.unsqueeze(0)
    mul_mask_vb = Variable(torch.zeros(vb.size(2), vb.size(1), vb.size(2))).type_as(vb)
    for i in range(vb.size(2)):
       mul_mask_vb[i, :, :i+1] = 1
    add_mask_vb = 1 - mul_mask_vb
    vb = vb.expand_as(mul_mask_vb) * mul_mask_vb + add_mask_vb
    # vb = torch.prod(vb, 2).transpose(0, 2)                # 0.1.12
    vb = torch.prod(vb, 2, keepdim=True).transpose(0, 2)    # 0.2.0
    # print(real_cumprod - vb.data) # NOTE: checked, ==0
    return vb


def batch_cosine_sim(u, v, epsilon=1e-6):
    """
    u: content_key: [batch_size x num_heads x mem_wid]
    v: memory:      [batch_size x mem_hei   x mem_wid]
    k: similarity:  [batch_size x num_heads x mem_hei]
    """
    assert u.dim() == 3 and v.dim() == 3
    numerator = torch.bmm(u, v.transpose(1, 2))
    # denominator = torch.sqrt(torch.bmm(u.norm(2, 2).pow(2) + epsilon, v.norm(2, 2).pow(2).transpose(1, 2) + epsilon))                             # 0.1.12
    denominator = torch.sqrt(torch.bmm(u.norm(2, 2, keepdim=True).pow(2) + epsilon, v.norm(2, 2, keepdim=True).pow(2).transpose(1, 2) + epsilon))   # 0.2.0
    k = numerator / (denominator + epsilon)
    return k



class ExternalMemory(object):
    def __init__(self, args):
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        self.batch_size = args.batch_size
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid

    def _save_memory(self):
        raise NotImplementedError("not implemented in base calss")

    def _load_memory(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self):
        self.memory_vb = Variable(self.memory_ts).type(self.dtype)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self.memory_ts = torch.zeros(self.batch_size, self.mem_hei, self.mem_wid).fill_(1e-6)
        self._reset_states()


class External2DMemory(ExternalMemory):
    def __init__(self, args):
        super(External2DMemory, self).__init__(args)
        self._reset()


class External3DMemory(ExternalMemory):
    def __init__(self, args):
        super(External3DMemory, self).__init__(args)
        self._reset()


class Head(nn.Module):
    def __init__(self, args):
        super(Head, self).__init__()
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.num_allowed_shifts = args.num_allowed_shifts

    def _reset_states(self):
        self.wl_prev_vb = Variable(self.wl_prev_ts).type(self.dtype) # batch_size x num_heads x mem_hei

    def _reset(self):           # NOTE: should be called at each child's __init__
        # self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # TODO: see how this one is reset
        # reset internal states
        self.wl_prev_ts = torch.eye(1, self.mem_hei).unsqueeze(0).expand(self.batch_size, self.num_heads, self.mem_hei)
        self._reset_states()

    def _access(self, memory_vb):
        # NOTE: called at the end of forward, to use the weight to read/write from/to memory
        raise NotImplementedError("not implemented in base calss")


class StaticHead(Head):
    def __init__(self, args):
        super(StaticHead, self).__init__(args)

        # build model
        # for content focus
        self.hid_2_key   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta  = nn.Linear(self.hidden_dim, self.num_heads * 1)
        # for location focus
        self.hid_2_gate  = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_shift = nn.Linear(self.hidden_dim, self.num_heads * self.num_allowed_shifts)
        self.hid_2_gamma = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)  # [batch_size x num_heads x mem_hei]
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb) # [batch_size x num_heads x mem_hei]
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2)).transpose(0, 2)

    def _shift(self, wg_vb, shift_vb):
        """
        variables needed:
            wg_vb:    [batch_size x num_heads x mem_hei]
            shift_vb: [batch_size x num_heads x num_allowed_shifts]
                   -> sum=1; the shift weight vector
        returns:
            ws_vb:    [batch_size x num_heads x mem_hei]
                   -> the attention weight by location focus
        """
        batch_size = wg_vb.size(0)
        input_dim = wg_vb.size(2); assert input_dim == self.mem_hei
        filter_dim = shift_vb.size(2); assert filter_dim == self.num_allowed_shifts

        ws_vb = None
        for i in range(batch_size): # for each head in each batch, the kernel is different ... seems there's no other way by doing the loop here
            for j in range(self.num_heads):
                ws_tmp_vb = F.conv1d(wg_vb[i][j].unsqueeze(0).unsqueeze(0).repeat(1, 1, 3),
                                     shift_vb[i][j].unsqueeze(0).unsqueeze(0).contiguous(),
                                     padding=filter_dim//2)[:, :, input_dim:(2*input_dim)]
                if ws_vb is None:
                    ws_vb = ws_tmp_vb
                else:
                    ws_vb = torch.cat((ws_vb, ws_tmp_vb), 0)
        ws_vb = ws_vb.view(-1, self.num_heads, self.mem_hei)
        return ws_vb

    def _location_focus(self):
        """
        variables needed:
            wl_prev_vb: [batch_size x num_heads x mem_hei]
            wc_vb:      [batch_size x num_heads x mem_hei]
            gate_vb:    [batch_size x num_heads x 1]
                     -> \in (0, 1); the interpolation gate
            shift_vb:   [batch_size x num_heads x num_allowed_shifts]
                     -> sum=1; the shift weight vector
            gamma_vb:   [batch_size x num_heads x 1]
                     -> >=1; the sharpening vector
        returns:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
                     -> the attention weight by location focus
        """
        self.gate_vb = self.gate_vb.expand_as(self.wc_vb)
        wg_vb = self.wc_vb * self.gate_vb + self.wl_prev_vb * (1. - self.gate_vb)
        ws_vb = self._shift(wg_vb, self.shift_vb)
        wp_vb = ws_vb.pow(self.gamma_vb.expand_as(ws_vb))
        # self.wl_curr_vb = wp_vb / wp_vb.sum(2).expand_as(wp_vb)               # 0.1.12
        self.wl_curr_vb = wp_vb / wp_vb.sum(2, keepdim=True).expand_as(wp_vb)   # 0.2.0

    def forward(self, hidden_vb, memory_vb):
        # outputs for computing addressing for heads
        # NOTE: to be consistent w/ the dnc paper, we use
        # NOTE: sigmoid to constrain to [0, 1]
        # NOTE: oneplus to constrain to [1, +inf]
        self.key_vb   = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)    # TODO: relu to bias the memory to store positive values ??? check again
        self.beta_vb  = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)          # beta >=1: https://github.com/deepmind/dnc/issues/9
        self.gate_vb  = F.sigmoid(self.hid_2_gate(hidden_vb)).view(-1, self.num_heads, 1)           # gate /in (0, 1): interpolation gate, blend wl_{t-1} & wc
        self.shift_vb = F.softmax(self.hid_2_shift(hidden_vb).view(-1, self.num_heads, self.num_allowed_shifts).transpose(0, 2)).transpose(0, 2)    # shift: /sum=1
        self.gamma_vb = (1. + F.softplus(self.hid_2_gamma(hidden_vb))).view(-1, self.num_heads, 1)  # gamma >= 1: sharpen the final weights

        # now we compute the addressing mechanism
        self._content_focus(memory_vb)
        self._location_focus()


class StaticReadHead(StaticHead):
    def __init__(self, args):
        super(StaticReadHead, self).__init__(args)
        # reset
        self._reset()

    def forward(self, hidden_vb, memory_vb):
        # content & location focus
        super(StaticReadHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        return self._access(memory_vb)

    def _access(self, memory_vb): # read
        """
        variables needed:
            wl_curr_vb:   [batch_size x num_heads x mem_hei]
                       -> location focus of {t}
            memory_vb:    [batch_size x mem_hei   x mem_wid]
        returns:
            read_vec_vb:  [batch_size x num_heads x mem_wid]
                       -> read vector of {t}
        """
        return torch.bmm(self.wl_curr_vb, memory_vb)


class StaticWriteHead(StaticHead):
    def __init__(self, args):
        super(StaticWriteHead, self).__init__(args)
        # buid model: add additional outs for write
        # for access
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)

        # reset
        self._reset()

    def forward(self, hidden_vb, memory_vb):
        # content & location focus
        super(StaticWriteHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb   = F.tanh(self.hid_2_add(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _access(self, memory_vb): # write
        """
        variables needed:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
            erase_vb:   [batch_size x num_heads x mem_wid]
                     -> /in (0, 1)
            add_vb:     [batch_size x num_heads x mem_wid]
                     -> w/ no restrictions in range
            memory_vb:  [batch_size x mem_hei x mem_wid]
        returns:
            memory_vb:  [batch_size x mem_hei x mem_wid]
        NOTE: IMPORTANT: https://github.com/deepmind/dnc/issues/10
        """

        # first let's do erasion
        weighted_erase_vb = torch.bmm(self.wl_curr_vb.contiguous().view(-1, self.mem_hei, 1),
                                      self.erase_vb.contiguous().view(-1, 1, self.mem_wid)).view(-1, self.num_heads, self.mem_hei, self.mem_wid)
        keep_vb = torch.prod(1. - weighted_erase_vb, dim=1)
        memory_vb = memory_vb * keep_vb
        # finally let's write (do addition)
        return memory_vb + torch.bmm(self.wl_curr_vb.transpose(1, 2), self.add_vb)


class DynamicHead(Head):
    def __init__(self, args):
        super(DynamicHead, self).__init__(args)

        # build model
        # for content focus
        self.hid_2_key  = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _update_usage(self, prev_usage_vb):
        raise NotImplementedError("not implemented in base calss")

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)  # [batch_size x num_heads x mem_hei]
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb) # [batch_size x num_heads x mem_hei]
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2), dim=0).transpose(0, 2)

    def _location_focus(self):
        raise NotImplementedError("not implemented in base calss")

    def forward(self, hidden_vb, memory_vb):
        # outputs for computing addressing for heads
        # NOTE: to be consistent w/ the dnc paper, we use
        # NOTE: sigmoid to constrain to [0, 1]
        # NOTE: oneplus to constrain to [1, +inf]
        self.key_vb   = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)    # TODO: relu to bias the memory to store positive values ??? check again
        self.beta_vb  = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)          # beta >=1: https://github.com/deepmind/dnc/issues/9

        # now we compute the addressing mechanism
        self._content_focus(memory_vb)


class DynamicReadHead(DynamicHead):
    def __init__(self, args):
        super(DynamicReadHead, self).__init__(args)
        # dynamic-accessor-specific params
        # NOTE: mixing between backwards and forwards position (for each write head)
        # NOTE: and content-based lookup, for each read head
        self.num_read_modes = args.num_read_modes

        # build model: add additional outs for read
        # for locatoin focus: temporal linkage
        self.hid_2_free_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_read_mode = nn.Linear(self.hidden_dim, self.num_heads * self.num_read_modes)

        # reset
        self._reset()

    def _update_usage(self, hidden_vb, prev_usage_vb):
        """
        calculates the new usage after reading and freeing from memory
        variables needed:
            hidden_vb:     [batch_size x hidden_dim]
            prev_usage_vb: [batch_size x mem_hei]
            free_gate_vb:  [batch_size x num_heads x 1]
            wl_prev_vb:    [batch_size x num_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        self.free_gate_vb = F.sigmoid(self.hid_2_free_gate(hidden_vb)).view(-1, self.num_heads, 1)
        free_read_weights_vb = self.free_gate_vb.expand_as(self.wl_prev_vb) * self.wl_prev_vb
        psi_vb = torch.prod(1. - free_read_weights_vb, 1)
        return prev_usage_vb * psi_vb

    def _directional_read_weights(self, link_vb, num_write_heads, forward):
        """
        calculates the forward or the backward read weights
        for each read head (at a given address), there are `num_writes` link
        graphs to follow. thus this function computes a read address for each of
        the `num_reads * num_writes` pairs of read and write heads.
        we calculate the forward and backward directions for each pair of read
        and write heads; hence we need to tile the read weights and do a sort of
        "outer product" to get this.
        variables needed:
            link_vb:    [batch_size x num_read_heads x mem_hei x mem_hei]
                     -> {L_t}, current link graph
            wl_prev_vb: [batch_size x num_read_heads x mem_hei]
                     -> containing the previous read weights w_{t-1}^r.
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            forward:    boolean
                     -> indicating whether to follow the "future" (True)
                     -> direction in the link graph or the "past" (False)
                     -> direction
        returns:
            directional_weights_vb: [batch_size x num_read_heads x num_write_heads x mem_hei]
        """
        expanded_read_weights_vb = self.wl_prev_vb.unsqueeze(1).expand_as(torch.Tensor(self.batch_size, num_write_heads, self.num_heads, self.mem_hei)).contiguous()
        if forward:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei).transpose(1, 2))
        else:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei))
        return directional_weights_vb.view(-1, num_write_heads, self.num_heads, self.mem_hei).transpose(1, 2)

    def _location_focus(self, link_vb, num_write_heads): # temporal linkage
        """
        calculates the read weights after location focus
        variables needed:
            link_vb:      [batch_size x num_heads x mem_hei x mem_hei]
                       -> {L_t}, current link graph
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            wc_vb:        [batch_size x num_heads x mem_hei]
                       -> containing the focus by content of {t}
            read_mode_vb: [batch_size x num_heads x num_read_modes]
        returns:
            wl_curr_vb:   [batch_size x num_read_heads x num_write_heads x mem_hei]
                       -> focus by content of {t}
        """
        # calculates f_t^i & b_t^i
        forward_weights_vb  = self._directional_read_weights(link_vb, num_write_heads, True)
        backward_weights_vb = self._directional_read_weights(link_vb, num_write_heads, False)
        backward_mode_vb = self.read_mode_vb[:, :, :num_write_heads]
        forward_mode_vb  = self.read_mode_vb[:, :, num_write_heads:2*num_write_heads]
        content_mode_vb  = self.read_mode_vb[:, :, 2*num_write_heads:]
        self.wl_curr_vb = content_mode_vb.expand_as(self.wc_vb) * self.wc_vb\
                        + torch.sum(forward_mode_vb.unsqueeze(3).expand_as(forward_weights_vb) * forward_weights_vb, 2) \
                        + torch.sum(backward_mode_vb.unsqueeze(3).expand_as(backward_weights_vb) * backward_weights_vb, 2)

    def _access(self, memory_vb): # read
        """
        variables needed:
            wl_curr_vb:   [batch_size x num_heads x mem_hei]
                       -> location focus of {t}
            memory_vb:    [batch_size x mem_hei   x mem_wid]
        returns:
            read_vec_vb:  [batch_size x num_heads x mem_wid]
                       -> read vector of {t}
        """
        return torch.bmm(self.wl_curr_vb, memory_vb)

    def forward(self, hidden_vb, memory_vb, link_vb, num_write_heads):
        # content focus
        super(DynamicReadHead, self).forward(hidden_vb, memory_vb)
        # location focus
        self.read_mode_vb = F.softmax(self.hid_2_read_mode(hidden_vb).view(-1, self.num_heads, self.num_read_modes).transpose(0, 2), dim=0).transpose(0, 2)
        self._location_focus(link_vb, num_write_heads)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        return self._access(memory_vb)


class DynamicWriteHead(DynamicHead):
    def __init__(self, args):
        super(DynamicWriteHead, self).__init__(args)

        # buid model: add additional outs for write
        # for location focus: dynamic allocation
        self.hid_2_alloc_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_write_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        # for access
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid) # the write vector in dnc, called "add" in ntm

        # reset
        self._reset()

    def _update_usage(self, prev_usage_vb):
        """
        calculates the new usage after writing to memory
        variables needed:
            prev_usage_vb: [batch_size x mem_hei]
            wl_prev_vb:    [batch_size x num_write_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        # calculate the aggregated effect of all write heads
        # NOTE: how multiple write heads are delt w/ is not discussed in the paper
        # NOTE: this part is only shown in the source code
        write_weights_vb = 1. - torch.prod(1. - self.wl_prev_vb, 1)
        return prev_usage_vb + (1. - prev_usage_vb) * write_weights_vb

    def _allocation(self, usage_vb, epsilon=1e-6):
        """
        computes allocation by sorting usage, a = a_t[\phi_t[j]]
        variables needed:
            usage_vb: [batch_size x mem_hei]
                   -> indicating current memory usage, this is equal to u_t in
                      the paper when we only have one write head, but for
                      multiple write heads, one should update the usage while
                      iterating through the write heads to take into account the
                      allocation returned by this function
        returns:
            alloc_vb: [batch_size x num_write_heads x mem_hei]
        """
        # ensure values are not too small prior to cumprod
        usage_vb = epsilon + (1 - epsilon) * usage_vb
        # NOTE: we sort usage in ascending order
        sorted_usage_vb, indices_vb = torch.topk(usage_vb, k=self.mem_hei, dim=1, largest=False)
        # to imitate tf.cumrprod(exclusive=True) https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        cat_sorted_usage_vb = torch.cat((Variable(torch.ones(self.batch_size, 1)).type(self.dtype), sorted_usage_vb), 1)[:, :-1]
        # TODO: seems we have to wait for this PR: https://github.com/pytorch/pytorch/pull/1439
        prod_sorted_usage_vb = fake_cumprod(cat_sorted_usage_vb)
        # prod_sorted_usage_vb = torch.cumprod(cat_sorted_usage_vb, dim=1) # TODO: use this once the PR is ready
        # alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb  # equ. (1)            # 0.1.12
        alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb.squeeze()  # equ. (1)    # 0.2.0
        _, indices_vb = torch.topk(indices_vb, k=self.mem_hei, dim=1, largest=False)
        alloc_weight_vb = alloc_weight_vb.gather(1, indices_vb)
        return alloc_weight_vb

    def _location_focus(self, usage_vb):
        """
        Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for
        each write head. (For more than one write head, we use a "simulated new
        usage" which takes into account the fact that the previous write head
        will increase the usage in that area of the memory.)
        variables needed:
            usage_vb:         [batch_size x mem_hei]
                           -> representing current memory usage
            write_gate_vb:    [batch_size x num_write_heads x 1]
                           -> /in [0, 1] indicating how much each write head
                              does writing based on the address returned here
                              (and hence how much usage increases)
        returns:
            alloc_weights_vb: [batch_size x num_write_heads x mem_hei]
                            -> containing the freeness-based write locations
                               Note that this isn't scaled by `write_gate`;
                               this scaling must be applied externally.
        """
        alloc_weights_vb = []
        for i in range(self.num_heads):
            alloc_weights_vb.append(self._allocation(usage_vb))
            # update usage to take into account writing to this new allocation
            # NOTE: checked: if not operate directly on _vb.data, then the _vb
            # NOTE: outside of this func will not change
            usage_vb += (1 - usage_vb) * self.write_gate_vb[:, i, :].expand_as(usage_vb) * alloc_weights_vb[i]
        # pack the allocation weights for write heads into one tensor
        alloc_weight_vb = torch.stack(alloc_weights_vb, dim=1)
        self.wl_curr_vb = self.write_gate_vb.expand_as(alloc_weight_vb) * (self.alloc_gate_vb.expand_as(self.wc_vb) * alloc_weight_vb + \
                                                                           (1. - self.alloc_gate_vb.expand_as(self.wc_vb)) * self.wc_vb)

    def _access(self, memory_vb): # write
        """
        variables needed:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
            erase_vb:   [batch_size x num_heads x mem_wid]
                     -> /in (0, 1)
            add_vb:     [batch_size x num_heads x mem_wid]
                     -> w/ no restrictions in range
            memory_vb:  [batch_size x mem_hei x mem_wid]
        returns:
            memory_vb:  [batch_size x mem_hei x mem_wid]
        NOTE: IMPORTANT: https://github.com/deepmind/dnc/issues/10
        """

        # first let's do erasion
        weighted_erase_vb = torch.bmm(self.wl_curr_vb.contiguous().view(-1, self.mem_hei, 1),
                                      self.erase_vb.contiguous().view(-1, 1, self.mem_wid)).view(-1, self.num_heads, self.mem_hei, self.mem_wid)
        keep_vb = torch.prod(1. - weighted_erase_vb, dim=1)
        memory_vb = memory_vb * keep_vb
        # finally let's write (do addition)
        return memory_vb + torch.bmm(self.wl_curr_vb.transpose(1, 2), self.add_vb)

    def forward(self, hidden_vb, memory_vb, usage_vb):
        # content focus
        super(DynamicWriteHead, self).forward(hidden_vb, memory_vb)
        # location focus
        self.alloc_gate_vb = F.sigmoid(self.hid_2_alloc_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self.write_gate_vb = F.sigmoid(self.hid_2_write_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self._location_focus(usage_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb   = F.tanh(self.hid_2_add(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _update_link(self, prev_link_vb, prev_preced_vb):
        """
        calculates the new link graphs
        For each write head, the link is a directed graph (represented by a
        matrix with entries in range [0, 1]) whose vertices are the memory
        locations, and an edge indicates temporal ordering of writes.
        variables needed:
            prev_link_vb:   [batch_size x num_heads x mem_hei x mem_wid]
                         -> {L_t-1}, previous link graphs
            prev_preced_vb: [batch_size x num_heads x mem_hei]
                         -> {p_t}, the previous aggregated precedence
                         -> weights for each write head
            wl_curr_vb:     [batch_size x num_heads x mem_hei]
                         -> location focus of {t}
        returns:
            link_vb:        [batch_size x num_heads x mem_hei x mem_hei]
                         -> {L_t}, current link graph
        """
        write_weights_i_vb = self.wl_curr_vb.unsqueeze(3).expand_as(prev_link_vb)
        write_weights_j_vb = self.wl_curr_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_preced_j_vb = prev_preced_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_link_scale_vb = 1 - write_weights_i_vb - write_weights_j_vb
        new_link_vb = write_weights_i_vb * prev_preced_j_vb
        link_vb = prev_link_scale_vb * prev_link_vb + new_link_vb
        # Return the link with the diagonal set to zero, to remove self-looping edges.
        # TODO: set diag as 0 if there's a specific method to do that w/ later releases
        diag_mask_vb = Variable(1 - torch.eye(self.mem_hei).unsqueeze(0).unsqueeze(0).expand_as(link_vb)).type(self.dtype)
        link_vb = link_vb * diag_mask_vb
        return link_vb

    def _update_precedence_weights(self, prev_preced_vb):
        """
        calculates the new precedence weights given the current write weights
        the precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the
        precedence weights unchanged, but with sum close to one will replace the
        precedence weights.
        variables needed:
            prev_preced_vb: [batch_size x num_write_heads x mem_hei]
            wl_curr_vb:     [batch_size x num_write_heads x mem_hei]
        returns:
            preced_vb:      [batch_size x num_write_heads x mem_hei]
        """
        # write_sum_vb = torch.sum(self.wl_curr_vb, 2)              # 0.1.12
        write_sum_vb = torch.sum(self.wl_curr_vb, 2, keepdim=True)  # 0.2.0
        return (1 - write_sum_vb).expand_as(prev_preced_vb) * prev_preced_vb + self.wl_curr_vb

    def _temporal_link(self, prev_link_vb, prev_preced_vb):
        link_vb = self._update_link(prev_link_vb, prev_preced_vb)
        preced_vb = self._update_precedence_weights(prev_preced_vb)
        return link_vb, preced_vb


class Accessor(nn.Module):
    def __init__(self, args):
        super(Accessor, self).__init__()
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # params
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

        # functional components
        self.write_head_params = args.write_head_params
        self.read_head_params = args.read_head_params
        self.memory_params = args.memory_params

        # fill in the missing values
        # write_heads
        self.write_head_params.num_heads = self.num_write_heads
        self.write_head_params.batch_size = self.batch_size
        self.write_head_params.hidden_dim = self.hidden_dim
        self.write_head_params.mem_hei = self.mem_hei
        self.write_head_params.mem_wid = self.mem_wid
        # read_heads
        self.read_head_params.num_heads = self.num_read_heads
        self.read_head_params.batch_size = self.batch_size
        self.read_head_params.hidden_dim = self.hidden_dim
        self.read_head_params.mem_hei = self.mem_hei
        self.read_head_params.mem_wid = self.mem_wid
        # memory
        self.memory_params.batch_size = self.batch_size
        self.memory_params.clip_value = self.clip_value
        self.memory_params.mem_hei = self.mem_hei
        self.memory_params.mem_wid = self.mem_wid

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset(self):           # NOTE: should be called at each child's __init__
        raise NotImplementedError("not implemented in base calss")

    def visual(self):
        self.write_heads.visual()
        self.read_heads.visual()
        self.memory.visual()

    def forward(self, lstm_hidden_vb):
        raise NotImplementedError("not implemented in base calss")


class StaticAccessor(Accessor):
    def __init__(self, args):
        super(StaticAccessor, self).__init__(args)
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # functional components
        self.write_heads = StaticWriteHead(self.write_head_params)
        self.read_heads = StaticReadHead(self.read_head_params)
        self.memory = External2DMemory(self.memory_params)

        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        # we reset the write/read weights of heads
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        # we also reset the memory to bias value
        self.memory._reset_states()

    def _reset(self):           # NOTE: should be called at __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self._reset_states()

    def forward(self, hidden_vb):
        # 1. first write to memory_{t-1} to get memory_{t}
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb)
        # 2. then read from memory_{t} to get read_vec_{t}
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb)
        return read_vec_vb


class DynamicAccessor(Accessor):
    def __init__(self, args):
        super(DynamicAccessor, self).__init__(args)
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # dynamic-accessor-specific params
        self.read_head_params.num_read_modes = self.write_head_params.num_heads * 2 + 1

        # functional components
        self.usage_vb = None    # for dynamic allocation, init in _reset
        self.link_vb = None     # for temporal link, init in _reset
        self.preced_vb = None   # for temporal link, init in _reset
        self.write_heads = DynamicWriteHead(self.write_head_params)
        self.read_heads = DynamicReadHead(self.read_head_params)
        self.memory = External2DMemory(self.memory_params)

        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        # reset the usage (for dynamic allocation) & link (for temporal link)
        self.usage_vb  = Variable(self.usage_ts).type(self.dtype)
        self.link_vb   = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)
        # we reset the write/read weights of heads
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        # we also reset the memory to bias value
        self.memory._reset_states()

    def _reset(self):           # NOTE: should be called at __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self.usage_ts  = torch.zeros(self.batch_size, self.mem_hei)
        self.link_ts   = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)
        self._reset_states()

    def forward(self, hidden_vb):
        # 1. first we update the usage using the read/write weights from {t-1}
        self.usage_vb = self.write_heads._update_usage(self.usage_vb)
        self.usage_vb = self.read_heads._update_usage(hidden_vb, self.usage_vb)
        # 2. then write to memory_{t-1} to get memory_{t}
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb, self.usage_vb)
        # 3. then we update the temporal link
        self.link_vb, self.preced_vb = self.write_heads._temporal_link(self.link_vb, self.preced_vb)
        # 4. then read from memory_{t} to get read_vec_{t}
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb, self.link_vb, self.write_head_params.num_heads)
        return read_vec_vb


class Controller(nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()
        # general params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.read_vec_dim = args.read_vec_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self):
        # we reset controller's hidden state
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_ts[0]).type(self.dtype),
                               Variable(self.lstm_hidden_ts[1]).type(self.dtype))

    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self.lstm_hidden_ts = []
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.hidden_dim))
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.hidden_dim))
        self._reset_states()

    def forward(self, input_vb):
        raise NotImplementedError("not implemented in base calss")


class LSTMController(Controller):
    def __init__(self, args):
        super(LSTMController, self).__init__(args)

        # build model
        self.in_2_hid = nn.LSTMCell(self.input_dim + self.read_vec_dim, self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input_vb, read_vec_vb):
        self.lstm_hidden_vb = self.in_2_hid(torch.cat((input_vb.contiguous().view(-1, self.input_dim),
                                                       read_vec_vb.contiguous().view(-1, self.read_vec_dim)), 1),
                                            self.lstm_hidden_vb)

        # we clip the controller hidden states here
        self.lstm_hidden_vb = [self.lstm_hidden_vb[0].clamp(min=-self.clip_value, max=self.clip_value),
                               self.lstm_hidden_vb[1]]

        return self.lstm_hidden_vb[0]


class Circuit(nn.Module):   # NOTE: basically this whole module is treated as a custom rnn cell
    def __init__(self, args):
        super(Circuit, self).__init__()
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

        # functional components
        self.controller_params = args.controller_params
        self.accessor_params = args.accessor_params

        # now we fill in the missing values for each module
        self.read_vec_dim = self.num_read_heads * self.mem_wid
        # controller
        self.controller_params.batch_size = self.batch_size
        self.controller_params.input_dim = self.input_dim
        self.controller_params.read_vec_dim = self.read_vec_dim
        self.controller_params.output_dim = self.output_dim
        self.controller_params.hidden_dim = self.hidden_dim
        self.controller_params.mem_hei = self.mem_hei
        self.controller_params.mem_wid = self.mem_wid
        self.controller_params.clip_value = self.clip_value
        # accessor: {write_heads, read_heads, memory}
        self.accessor_params.batch_size = self.batch_size
        self.accessor_params.hidden_dim = self.hidden_dim
        self.accessor_params.num_write_heads = self.num_write_heads
        self.accessor_params.num_read_heads = self.num_read_heads
        self.accessor_params.mem_hei = self.mem_hei
        self.accessor_params.mem_wid = self.mem_wid
        self.accessor_params.clip_value = self.clip_value

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self): # should be called at the beginning of forwarding a new input sequence
        # we first reset the previous read vector
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        # we then reset the controller's hidden state
        self.controller._reset_states()
        # we then reset the write/read weights of heads
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        # reset internal states
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim).fill_(1e-6)
        self._reset_states()

    def forward(self, input_vb):
        # NOTE: the operation order must be the following: control, access{write, read}, output

        # 1. first feed {input, read_vec_{t-1}} to controller
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        # 2. then we write to memory_{t-1} to get memory_{t}; then read from memory_{t} to get read_vec_{t}
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        # 3. finally we concat the output from the controller and the current read_vec_{t} to get the final output
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.hidden_dim),
                                               self.read_vec_vb.view(-1, self.read_vec_dim)), 1))

        # we clip the output values here
        return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(1, self.batch_size, self.output_dim)


class NTMCircuit(Circuit):
    def __init__(self, args):
        super(NTMCircuit, self).__init__(args)

        # functional components
        self.controller = LSTMController(self.controller_params)
        self.accessor = StaticAccessor(self.accessor_params)

        # build model
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)

        self._reset()

    def _init_weights(self):
        pass


class DNCCircuit(Circuit):
    def __init__(self, args):
        super(DNCCircuit, self).__init__(args)

        # functional components
        self.controller = LSTMController(self.controller_params)
        self.accessor = DynamicAccessor(self.accessor_params)

        # build model
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)

        self._reset()

    def _init_weights(self):
        pass


if __name__ == "__main__":
    class Empty():
        use_cuda = False
        dtype = torch.FloatTensor


    ntm_args = Empty()
    ntm_args.batch_size = 16
    ntm_args.input_dim = 10
    ntm_args.output_dim = 8
    ntm_args.hidden_dim = 100
    ntm_args.num_write_heads = 1
    ntm_args.num_read_heads = 1
    ntm_args.mem_hei = 128
    ntm_args.mem_wid = 20
    ntm_args.clip_value = 20.
    ntm_args.controller_params = Empty()
    ntm_args.accessor_params = Empty()
    ntm_args.accessor_params.write_head_params = Empty()
    ntm_args.accessor_params.write_head_params.num_allowed_shifts = 3
    ntm_args.accessor_params.read_head_params = Empty()
    ntm_args.accessor_params.read_head_params.num_allowed_shifts = 3
    ntm_args.accessor_params.memory_params = Empty()
    ntm = NTMCircuit(ntm_args)


    dnc_args = Empty()
    dnc_args.batch_size = 16
    dnc_args.input_dim = 10
    dnc_args.output_dim = 8
    dnc_args.hidden_dim = 64
    dnc_args.num_write_heads = 1
    dnc_args.num_read_heads = 4
    dnc_args.mem_hei = 16
    dnc_args.mem_wid = 16
    dnc_args.clip_value = 20.
    dnc_args.controller_params = Empty()
    dnc_args.accessor_params = Empty()
    dnc_args.accessor_params.write_head_params = Empty()
    dnc_args.accessor_params.write_head_params.num_allowed_shifts = 3
    dnc_args.accessor_params.read_head_params = Empty()
    dnc_args.accessor_params.read_head_params.num_allowed_shifts = 3
    dnc_args.accessor_params.memory_params = Empty()
    dnc = DNCCircuit(dnc_args)
