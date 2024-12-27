import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, s_dim, a_dim, g_dim, batch_size):
        self.size = size
        self.s0_buf = np.zeros((size, s_dim), dtype=np.float32)
        self.a_buf = np.zeros((size, a_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size,1), dtype=np.float32)
        self.s1_buf = np.zeros((size, s_dim), dtype=np.float32)
        self.terminal_buf = np.zeros((size,1), dtype=np.float32)
        self.g0_buf = np.zeros((size, g_dim), dtype=np.float32)
        self.g1_buf = np.zeros((size, g_dim), dtype=np.float32)
        self.ptr = 0
        self.full = False
        self.batch_size = batch_size

        self.length_buf = 0

    def peek_buffer(self):
        return [self.s0_buf, self.g0_buf, self.a_buf, self.reward_buf, self.s1_buf, self.g1_buf, self.terminal_buf]

    def __len__(self) -> int:
        return self.length_buf

    def store(self, s0, g0, a, r, s1, g1, terminal):
        self.s0_buf[self.ptr] = s0
        self.g0_buf[self.ptr] = g0
        self.a_buf[self.ptr] = a
        self.reward_buf[self.ptr] = r
        self.s1_buf[self.ptr] = s1
        self.g1_buf[self.ptr] = g1
        self.terminal_buf[self.ptr] = terminal
        self.ptr += 1

        # Buffer Full
        if self.ptr == self.size:
            self.ptr = 0
            self.full = True
            self.length_buf = self.size
        if self.full == False:
            self.length_buf = self.ptr


    def store_many(self, s0, g0, a, r, s1, g1, terminal, length):
        if self.ptr + length >= self.size:
            firstpart = self.size-self.ptr
            secondpart = length - firstpart
            self.s0_buf[self.ptr:] = s0[:firstpart]
            self.s0_buf[:secondpart] = s0[firstpart:]
            
            self.g0_buf[self.ptr:] = g0[:firstpart]
            self.g0_buf[:secondpart] = g0[firstpart:]

            self.a_buf[self.ptr:] = a[:firstpart]
            self.a_buf[:secondpart] = a[firstpart:]

            self.reward_buf[self.ptr:] = r[:firstpart]
            self.reward_buf[:secondpart] = r[firstpart:]

            self.s1_buf[self.ptr:] = s1[:firstpart]
            self.s1_buf[:secondpart] = s1[firstpart:]

            self.g1_buf[self.ptr:] = g1[:firstpart]
            self.g1_buf[:secondpart] = g1[firstpart:]
            
            self.terminal_buf[self.ptr:] = terminal[:firstpart]
            self.terminal_buf[:secondpart] = terminal[firstpart:]

            self.ptr= secondpart
            self.full = True

        else:

            self.s0_buf[self.ptr: self.ptr+length] = s0
            self.g0_buf[self.ptr: self.ptr+length] = g0
            self.a_buf[self.ptr: self.ptr+length] = a
            self.reward_buf[self.ptr: self.ptr+length] = r
            self.s1_buf[self.ptr: self.ptr+length] = s1
            self.g1_buf[self.ptr: self.ptr+length] = g1
            self.terminal_buf[self.ptr: self.ptr+length] = terminal

            self.ptr += length

        if self.full:
            self.length_buf = self.size
        else:
            self.length_buf = self.ptr

    def _encode_sample(self, idxes):
        s0 = self.s0_buf[idxes]
        g0 = self.g0_buf[idxes]
        a = self.a_buf[idxes]
        r = self.reward_buf[idxes]
        s1 = self.s1_buf[idxes]
        g1 = self.g1_buf[idxes]
        terminal = self.terminal_buf[idxes]

        return [s0, g0, a, r, s1, g1, terminal]


    def sample(self):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        # print("size:", self.size, "ptr:", self.ptr)
        if self.full:
            idxes = [random.randint(0, self.size - 1) for _ in range(self.batch_size)]
        else:
            idxes = [random.randint(0, self.ptr - 1) for _ in range(self.batch_size)]
        return self._encode_sample(idxes)



class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

# class Transition():
#     def __init__(self, s0, g0, a, r, s1, g1, terminal):
#         self.s0 = s0
#         self.a = a
#         self.reward = r
#         self.s1 = s1
#         self.terminal = terminal
#         self.g0 = g0
#         self.g1 = g1

class Prioritized_ReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, size, s_dim, a_dim, g_dim, batch_size):
        self.tree = SumTree(size)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.g_dim = g_dim
        self.batch_size = batch_size
        self.size = size
        self.length = 0

    def store(self, s0, g0, a, r, s1, g1, terminal):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, [s0, g0, a, r, s1, g1, terminal])   # set the max p for new p
        if self.length < self.size:
            self.length += 1

    def __len__(self) -> int:
        return self.length
    
    def store_many(self, s0, g0, a, r, s1, g1, terminal, length):
        for a,b,c,d,e,f,g in zip(s0, g0, a, r, s1, g1, terminal):
            self.store(a,b,c,d,e,f,g)
    
    def sample(self):
        n = self.batch_size
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [None] * n, np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        i = 0
        while i < n:
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            if data == 0:
                continue
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
            i += 1
        return b_idx, b_memory, ISWeights

    def update_priorities(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    
        