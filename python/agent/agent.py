import os
import sys
from . import context
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import time
import numpy as np
import math

EXPLORE = 4000
STDDEV = 0.1
NSTEP = 0.3

from . import context
from helpers.buffer import ReplayBuffer, Prioritized_ReplayBuffer
from helpers.noise import OU_Noise, G_Noise,Random_Noise
from helpers.utils import create_input_op_shape


class Actor:
    def __init__(self, a_dim, h1_shape, h2_shape, action_scale=1.0, name="actor"):
        self.a_dim = a_dim
        self.name = name
        self.action_scale = action_scale
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape

    def train_var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build(self, s, is_training):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(s, units=self.h1_shape, name="fc1")
            h1 = tf.layers.batch_normalization(h1, training=is_training, scale=False)
            h1 = tf.nn.leaky_relu(h1)

            h2 = tf.layers.dense(h1, units=self.h2_shape, name="fc2")
            h2 = tf.layers.batch_normalization(h2, training=is_training, scale=False)
            h2 = tf.nn.leaky_relu(h2)

            h3 = tf.layers.dense(h2, units=math.ceil(self.h2_shape / 2), name="fc3")
            h3 = tf.layers.batch_normalization(h3, training=is_training, scale=False)
            h3 = tf.nn.leaky_relu(h3)

            # s = tf.Print(s, [s], summarize=100)
            output = tf.layers.dense(h3, units=self.a_dim, activation=tf.nn.tanh)
            # output = tf.Print(output, [output], summarize=100)
            scale_output = tf.multiply(output, self.action_scale)

        return scale_output


class Critic:
    def __init__(self, h1_shape, h2_shape, action_scale=1.0, name="critic"):
        self.name = name
        self.action_scale = action_scale
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape

    def train_var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build(self, s, action):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h1 = tf.layers.dense(
                s, units=self.h1_shape, activation=tf.nn.leaky_relu, name="fc1"
            )

            h2 = tf.layers.dense(
                tf.concat([h1, action], -1),
                units=self.h2_shape,
                activation=tf.nn.leaky_relu,
                name="fc2",
            )

            h3 = tf.layers.dense(
                h2,
                units=math.ceil(self.h2_shape / 2),
                activation=tf.nn.leaky_relu,
                name="fc3",
            )

            output = tf.layers.dense(h3, units=1)

        return output


def single_head_attention(z, dv, dk):
    """
    https://colab.research.google.com/github/zaidalyafeai/AttentioNN/blob/master/TransformerI.ipynb
    """
    # 1. projection
    # inp z: [None, n, dm]
    # out Q: [None, n, dk]  K: [None, n, dk] V: [None, n, dv]
    V = tf.keras.layers.Dense(units=dv)(z)
    Q = tf.keras.layers.Dense(units=dk)(z)
    K = tf.keras.layers.Dense(units=dk)(z)

    # 2. scaled dot product
    # inp Q:  [None, n, dk] K: [None, n, dk]
    # out score : [None, n, n]
    score = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(dk * 1.0)

    # 3. evaluate the weights
    # inp score: [None, n, n]
    # out W: [None, n, n]
    W = tf.nn.softmax(score)

    # 4. evaluate the context vector
    # inp W: [None, n, n] V: [None, n, dv]
    # out H: [None, n, dv]
    H = tf.matmul(W, V)
    return H


def multi_head_attention(z, head_num, dv, dk):
    Hs = []
    # 1. apply h times
    # inp z : [None, n, dm]
    # out Hs: [[None, n, dv], ..., [None, n, dv]]
    for i in range(0, head_num):
        # single head attention
        # inp z: [None, n, dm]
        # out H: [None, n, dv]
        H = single_head_attention(z, dv, dk)
        Hs.append(H)
    # 2. concatenate
    # inp Hs: [[None, n, dv], ..., [None, n, dv]]
    # out z : [None, n, dv * head_num] => [None, dout]
    z = tf.concat(Hs, axis=-1)
    z = tf.reduce_sum(z, axis=1)
    # z = tf.keras.layers.Dense(units = dout)(tf.keras.layers.Flatten()(z))
    return z


class Agent:
    def __init__(
        self,
        s_dim,
        s_dim_global,
        a_dim,
        h1_shape,
        h2_shape,
        gamma=0.995,
        batch_size=8,
        lr_a=1e-4,
        lr_c=1e-3,
        tau=1e-3,
        policy_delay=5,
        mem_size=1e5,
        action_scale=1.0,
        action_range=(-1.0, 1.0),
        noise_type=3,
        noise_exp=50000,
        train_exp=500000,
        summary=None,
        stddev=0.1,
        PER=False,
        alpha=0.6,
        LOSS_TYPE="HUBER",
        train_dir="./train_dir",
        eval_dir="./eval_dir",
        ckpt_dir="./ckpt_dir",
        is_global=False,
        max_flow_num=5,
    ):
        self.PER = PER
        self.LOSS_TYPE = LOSS_TYPE
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.s_dim = s_dim
        self.is_global = is_global
        self.a_dim = a_dim
        self.gamma = gamma
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.ckpt_dir = ckpt_dir
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.s_dim_global = s_dim_global
        self.policy_delay = policy_delay
        self.s0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name="s0")
        self.s0_global = tf.placeholder(
            tf.float32, shape=[None, self.s_dim_global], name="s0_global"
        )
        self.s1 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name="s1")
        self.s1_global = tf.placeholder(
            tf.float32, shape=[None, self.s_dim_global], name="s1_global"
        )
        self.is_training = tf.placeholder(tf.bool, name="Actor_is_training")
        self.action = tf.placeholder(tf.float32, shape=[None, a_dim], name="action")
        self.lr_a_ph = tf.placeholder(tf.float32, [], name="actor_lr_ph")
        self.lr_c_ph = tf.placeholder(tf.float32, [], name="critic_lr_ph")
        self.noise_type = noise_type
        self.noise_exp = noise_exp
        self.train_exp = train_exp
        self.action_range = action_range
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape
        self.stddev = stddev
        self.learn = True
        if not self.PER:
            self.rp_buffer = ReplayBuffer(
                int(mem_size), s_dim, a_dim, s_dim_global, batch_size=batch_size
            )
        else:
            self.rp_buffer = Prioritized_ReplayBuffer(
                int(mem_size), s_dim, a_dim, s_dim_global, batch_size=batch_size
            )

        if noise_type == 0:
            self.actor_noise = Random_Noise(
                explore=self.noise_exp
            )
        elif noise_type == 1:
            self.actor_noise = OU_Noise(
                mu=np.zeros(a_dim),
                sigma=float(self.stddev) * np.ones(a_dim),
                dt=1,
                exp=self.noise_exp,
            )
        elif noise_type == 2:
            ## Gaussian with gradually decay
            self.actor_noise = G_Noise(
                mu=np.zeros(a_dim),
                sigma=float(self.stddev) * np.ones(a_dim),
                explore=self.noise_exp,
            )
        elif noise_type == 3:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(
                mu=np.zeros(a_dim),
                sigma=float(self.stddev) * np.ones(a_dim),
                explore=None,
                theta=0.1,
            )
        elif noise_type == 4:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(
                mu=np.zeros(a_dim),
                sigma=float(self.stddev) * np.ones(a_dim),
                explore=EXPLORE,
                theta=0.1,
                mode="step",
                step=NSTEP,
            )
        elif noise_type == 5:
            self.actor_noise = None
        else:
            self.actor_noise = OU_Noise(
                mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), dt=0.5
            )

        # Main Actor/Critic Network
        self.actor = Actor(
            self.a_dim,
            action_scale=action_scale,
            h1_shape=self.h1_shape,
            h2_shape=self.h2_shape,
        )
        self.critic = Critic(
            action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape
        )
        self.critic2 = Critic(
            action_scale=action_scale,
            name="critic2",
            h1_shape=self.h1_shape,
            h2_shape=self.h2_shape,
        )
        self.actor_out = self.actor.build(self.s0, self.is_training)
        # global_s0 = multi_head_attention(self.s0_global, 1, 32, 16)
        # global_s1 = multi_head_attention(self.s1_global, 1, 32, 16)
        if self.is_global:
            q_input_0 = tf.concat([self.s0, self.s0_global], 1)
            q_input_1 = tf.concat([self.s1, self.s1_global], 1)
        else:
            q_input_0 = self.s0
            q_input_1 = self.s1

        self.critic_out = self.critic.build(q_input_0, self.action)
        self.critic_out2 = self.critic2.build(q_input_0, self.action)
        self.critic_actor_out = self.critic.build(q_input_0, self.actor_out)

        # Target Actor/Critic network
        self.target_actor = Actor(
            self.a_dim,
            action_scale=action_scale,
            h1_shape=self.h1_shape,
            h2_shape=self.h2_shape,
            name="target_actor",
        )
        self.target_critic = Critic(
            action_scale=action_scale,
            h1_shape=self.h1_shape,
            h2_shape=self.h2_shape,
            name="target_critic",
        )
        self.target_critic2 = Critic(
            action_scale=action_scale,
            name="target_critic2",
            h1_shape=self.h1_shape,
            h2_shape=self.h2_shape,
        )

        self.target_actor_out = self.target_actor.build(self.s1, self.is_training)
        self.target_actor_policy = self.get_target_actor_policy()
        self.target_critic_actor_out = self.target_critic.build(
            q_input_1, self.target_actor_policy
        )
        self.target_critic_actor_out2 = self.target_critic2.build(
            q_input_1, self.target_actor_policy
        )

        self.target_actor_update_op = self.target_update_op(
            self.target_actor.train_var(), self.actor.train_var(), tau
        )
        self.target_critic_update_op = self.target_update_op(
            self.target_critic.train_var(), self.critic.train_var(), tau
        )
        self.target_critic_update_op2 = self.target_update_op(
            self.target_critic2.train_var(), self.critic2.train_var(), tau
        )

        self.target_act_init_op = self.target_init(
            self.target_actor.train_var(), self.actor.train_var()
        )
        self.target_cri_init_op = self.target_init(
            self.target_critic.train_var(), self.critic.train_var()
        )
        self.target_cri_init_op2 = self.target_init(
            self.target_critic2.train_var(), self.critic2.train_var()
        )

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.terminal = tf.placeholder(tf.float32, shape=[None, 1], name="is_terminal")
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
        self.min_target_critic_actor_out = tf.minimum(
            self.target_critic_actor_out, self.target_critic_actor_out2
        )
        self.y = tf.stop_gradient(
            self.reward
            + self.gamma * (1 - self.terminal) * self.min_target_critic_actor_out
        )

        self.importance = tf.placeholder(
            tf.float32, [None, 1], name="imporance_weights"
        )
        self.td_error = self.critic_out - self.y

        self.summary_writer = summary
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

    def build_learn(self):
        self.learning_rate_a = tf.compat.v1.train.exponential_decay(
            self.lr_a, self.global_step, self.train_exp, 0.96, staircase=True
        )
        self.learning_rate_c = tf.compat.v1.train.exponential_decay(
            self.lr_c, self.global_step, self.train_exp, 0.96, staircase=True
        )
        # lr_schedule_c = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=self.lr_c,
        #     decay_steps=10000,
        #     decay_rate=0.96)
        self.actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_a
        )
        self.critic_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_c
        )

        self.actor_train_op = self.build_actor_train_op()
        self.critic_train_op = self.build_critic_train_op()

    def build_critic_train_op(self):
        def f1(y, pred, weights=1.0):
            error = tf.square(y - pred)
            weighted_error = tf.reduce_mean(error * weights)
            return weighted_error

        loss_function = {"HUBER": tf.compat.v1.losses.huber_loss, "MSE": f1}
        if self.PER:
            self.c_loss = loss_function[self.LOSS_TYPE](
                self.y, self.critic_out, weights=self.importance
            )
            self.c_loss2 = loss_function[self.LOSS_TYPE](
                self.y, self.critic_out2, weights=self.importance
            )
        else:
            self.c_loss = loss_function[self.LOSS_TYPE](self.y, self.critic_out)
            self.c_loss2 = loss_function[self.LOSS_TYPE](self.y, self.critic_out2)

        all_c_loss = self.c_loss + self.c_loss2
        critic_op = self.critic_optimizer.minimize(
            all_c_loss, var_list=self.critic.train_var() + self.critic2.train_var()
        )
        return critic_op

    def create_tf_summary(self):
        tf.summary.scalar("Loss/critic_loss:", self.c_loss)
        tf.summary.scalar("Loss/critic_loss_2:", self.c_loss2)
        tf.summary.scalar("Loss/actor_loss:", self.a_loss)
        tf.summary.scalar("Loss/lr_a:", self.learning_rate_a)

        self.summary_op = tf.summary.merge_all()

    def init_target(self):
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_act_init_op)
        self.sess.run(self.target_cri_init_op)
        self.sess.run(self.target_cri_init_op2)

    def get_target_actor_policy(self):
        # target policy smoothing
        eps = tf.random_normal(tf.shape(self.target_actor_out), stddev=0.05)
        eps = tf.clip_by_value(eps, -0.1, 0.1)
        t_a = self.target_actor_out + eps
        t_a = tf.clip_by_value(t_a, -1.0, 1.0)
        return t_a

    def assign_sess(self, sess):
        self.sess = sess

    def build_actor_train_op(self):
        self.a_loss = -tf.reduce_mean(self.critic_actor_out)
        # a_loss_output = tf.Print(self.a_loss, [self.a_loss], message="aloss:")
        return self.actor_optimizer.minimize(
            self.a_loss, var_list=self.actor.train_var(), global_step=self.global_step
        )

    def target_init(self, target, vars):
        return [tf.assign(target[i], vars[i]) for i in range(len(vars))]

    def target_update_op(self, target, vars, tau):
        return [
            tf.assign(target[i], vars[i] * tau + target[i] * (1 - tau))
            for i in range(len(vars))
        ]

    def target_update_hard_op(self, target, vars):
        return [tf.assign(target[i], vars[i]) for i in range(len(vars))]

    def actor_clone_update(self):
        self.sess.run(self.actor_clone_update_op)

    def get_action(self, s, use_noise=True):
        fd = {self.s0: create_input_op_shape(s, self.s0), self.is_training: False}
        action = self.sess.run([self.actor_out], feed_dict=fd)
        if use_noise:
            # print(self.actor_noise)
            noise = self.actor_noise(action[0])[0]
            # print("action: ", action, "noise:", noise)
            action += noise
            action = np.clip(action, self.action_range[0], self.action_range[1])
            # print("clipped action:", action, "noise:", noise)
        return action

    def get_q(self, s, a):

        fd = {
            self.s0: create_input_op_shape(s, self.s0),
            self.action: create_input_op_shape(a, self.action),
        }

        return self.sess.run([self.critic_out], feed_dict=fd)

    def get_q_actor(self, s):

        fd = {self.s0: create_input_op_shape(s, self.s0)}
        return self.sess.run([self.critic_actor_out], feed_dict=fd)

    def store_experience(self, s0, g0, a, r, s1, g1, terminal):
        self.rp_buffer.store(s0, g0, a, r, s1, g1, terminal)

    def store_many_experience(self, s0, g0, a, r, s1, g1, terminal, length):
        self.rp_buffer.store_many(s0, g0, a, r, s1, g1, terminal, length)

    def sample_experince(self):
        return self.rp_buffer.sample()

    def train_step(self, step):
        extra_update_ops = [
            v
            for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if "actor" in v.name and "target" not in v.name
        ]

        if self.PER == True:
            idxes, batch_sample, weights = self.rp_buffer.sample()
            batch_samples = list(zip(*batch_sample))
            fd = {
                self.s0: create_input_op_shape(batch_samples[0], self.s0),
                self.s0_global: create_input_op_shape(batch_samples[1], self.s0_global),
                self.action: create_input_op_shape(batch_samples[2], self.action),
                self.reward: create_input_op_shape(batch_samples[3], self.reward),
                self.s1: create_input_op_shape(batch_samples[4], self.s1),
                self.s1_global: create_input_op_shape(batch_samples[5], self.s1_global),
                self.terminal: create_input_op_shape(batch_samples[6], self.terminal),
                self.is_training: True,
                self.importance: weights,
                self.lr_a_ph: self.lr_a,
                self.lr_c_ph: self.lr_c,
            }

        else:
            batch_samples = self.rp_buffer.sample()
            fd = {
                self.s0: create_input_op_shape(batch_samples[0], self.s0),
                self.s0_global: create_input_op_shape(batch_samples[1], self.s0_global),
                self.action: create_input_op_shape(batch_samples[2], self.action),
                self.reward: create_input_op_shape(batch_samples[3], self.reward),
                self.s1: create_input_op_shape(batch_samples[4], self.s1),
                self.s1_global: create_input_op_shape(batch_samples[5], self.s1_global),
                self.terminal: create_input_op_shape(batch_samples[6], self.terminal),
                self.is_training: True,
                self.lr_a_ph: self.lr_a,
                self.lr_c_ph: self.lr_c,
            }
        # print("check fd:", fd)
        self.sess.run([self.critic_train_op], feed_dict=fd)
        self.sess.run(
            [
                self.target_critic_update_op,
                self.target_critic_update_op2,
            ],
            feed_dict=fd,
        )
        if step % self.policy_delay == 0:
            # print(step, "update policy")
            self.sess.run([self.actor_train_op, extra_update_ops], feed_dict=fd)
            self.sess.run(
                [
                    self.target_actor_update_op
                ],
                feed_dict=fd,
            )
            # print(self.sess.run([self.global_step]))
        if self.PER:
            td_errors = self.sess.run([self.td_error], feed_dict=fd)
            new_priorities = np.abs(np.squeeze(td_errors)) + 1e-6
            self.rp_buffer.update_priorities(idxes, new_priorities)

        summary = self.sess.run(self.summary_op, feed_dict=fd)
        self.summary_writer.add_summary(summary, global_step=step)

    def log_tf(self, val, tag=None, step_counter=0):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=val)
        self.summary_writer.add_summary(summary, step_counter)

    def save_model(self, step=None):
        self.saver.save(self.sess._sess._sess._sess._sess, os.path.join(self.ckpt_dir, 'model'), global_step =step)

    def load_model(self, name=None):
        if name is not None:
            print(os.path.join(self.ckpt_dir, name))
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ckpt_dir))
