import tensorflow as tf
import numpy as np
from predictors import ActorCriticTimeSeriesPredictor
from models import ActorCriticTransformer
import vtrace


EPOCH_COUNT = 1

LEARNING_RATE = 0.0003
CONVICTION_ACTION_COUNT = 2
CONVICTION_FEATURE_COUNT = 21
POSITION_ACTION_COUNT = 10
POSITION_FEATURE_COUNT = 7
SEQUENCE_LENGTH = 240
ATTENTION_HEAD_COUNT = 12
ATTENTION_DENSE_SIZE = 256
DENSE_SIZE = 256
FEED_FORWARD_DIMENTION = 256
DROPOUT_RATE = 0.1


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def network(state, num_action ):
     model_network = ActorCriticTransformer( num_action, CONVICTION_FEATURE_COUNT,
                    SEQUENCE_LENGTH, ATTENTION_HEAD_COUNT, ATTENTION_DENSE_SIZE, DENSE_SIZE, FEED_FORWARD_DIMENTION,
                    DROPOUT_RATE)
    critic , actor = model_network.call(state)
    return actor, critic

def build_network(state, previous_action, 
                  trajectory_state, trajectory_previous_action, 
                  num_action,  trajectory):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
        policy, _ = network(
            state=state,
            num_action=num_action)

    unrolled_first_state = trajectory_state[:, :-2]
    unrolled_middle_state = trajectory_state[:, 1:-1]
    unrolled_last_state = trajectory_state[:, 2:]

    unrolled_first_previous_action = trajectory_previous_action[:, :-2]
    unrolled_middle_previous_action = trajectory_previous_action[:, 1:-1]
    unrolled_last_previous_action = trajectory_previous_action[:, 2:]

    unrolled_first_policy = []
    unrolled_first_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                state=unrolled_first_state[:, i],
                num_action=num_action)
            unrolled_first_policy.append(p)
            unrolled_first_value.append(v)
    unrolled_first_policy = tf.stack(unrolled_first_policy, axis=1)
    unrolled_first_value = tf.stack(unrolled_first_value, axis=1)

    unrolled_middle_policy = []
    unrolled_middle_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                state=unrolled_middle_state[:, i]
                num_action=num_action)
            unrolled_middle_policy.append(p)
            unrolled_middle_value.append(v)
    unrolled_middle_policy = tf.stack(unrolled_middle_policy, axis=1)
    unrolled_middle_value = tf.stack(unrolled_middle_value, axis=1)

    unrolled_last_policy = []
    unrolled_last_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_last_state[:, i],
                num_action=num_action)
            unrolled_last_policy.append(p)
            unrolled_last_value.append(v)
    unrolled_last_policy = tf.stack(unrolled_last_policy, axis=1)
    unrolled_last_value = tf.stack(unrolled_last_value, axis=1)

    return policy,unrolled_first_policy, unrolled_first_value, \
        unrolled_middle_policy, unrolled_middle_value, \
            unrolled_last_policy, unrolled_last_value

class IMPALA:
    def __init__(self, trajectory, input_shape_conviction, input_shape_position , num_action_conviction , num_action_position , discount_factor, start_learning_rate,
                 end_learning_rate, learning_frame, baseline_loss_coef, entropy_coef, gradient_clip_norm,
                 reward_clipping, model_name, learner_name):

        self.input_shape_conviction = input_shape_conviction
        self.input_shape_position = input_shape_position
        self.trajectory = trajectory
        self.num_action_conviction = num_action_conviction
        self.num_action_position = num_action_position
        self.discount_factor = discount_factor
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm

        with tf.variable_scope(model_name):

            with tf.device('cpu'):

                self.state_placeholder_conviction_network = tf.placeholder(tf.float32, shape=[None, *self.input_shape_conviction])
                self.state_placeholder_position_network = tf.placeholder(tf.float32, shape=[None, *self.input_shape_position])
                self.conviction_previous_action_placeholder = tf.placeholder(tf.int32, shape=[None])
                self.position_previous_action_placeholder = tf.placeholder(tf.int32, shape=[None])

                self.trajectory_state_placeholder_conviction_network = tf.placeholder(tf.float32, shape=[None, self.trajectory, *self.input_shape_conviction])
                self.trajectory_state_placeholder_position_network = tf.placeholder(tf.float32, shape=[None, self.trajectory, *self.input_shape_position]
                self.trajectory_conviction_previous_action_placeholder = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.trajectory_position_previous_action_placeholder = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.conviction_action_placeholder = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.position_action_placeholder = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.reward_placeholder = tf.placeholder(tf.float32, shape=[None, self.trajectory])
                self.discount_placeholder = tf.placeholder(tf.bool, shape=[None, self.trajectory])
                self.conviction_network_behaviour_policy_placeholder = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.num_action_conviction])
                self.position_network_behaviour_policy_placeholder = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.num_action_position])

                if reward_clipping == 'abs_one':
                    self.clipped_reward_placeholder = tf.clip_by_value(self.reward_placeholder, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.reward_placeholder / 5.0)
                    self.clipped_reward_placeholder = tf.where(self.reward_placeholder < 0, .3 * squeezed, squeezed) * 5.

                self.discounts = tf.to_float(~self.discount_placeholder) * self.discount_factor

                self.conviction_policy, self.unrolled_conviction_first_policy, \
                    self.unrolled_conviction_first_value, self.unrolled_conviction_middle_policy,\
                        self.unrolled_conviction_middle_value, self.unrolled_conviction_last_policy,\
                            self.unrolled_conviction_last_value = build_network(
                                                        state=self.state_placeholder_conviction_network, previous_action=self.conviction_previous_action_placeholder, trajectory=self.trajectory,
                                                        num_action=self.num_action_conviction, 
                                                        trajectory_state=self.trajectory_state_placeholder_conviction_network, trajectory_previous_action=self.trajectory_conviction_previous_action_placeholder,
                                                        )

                self.position_policy, self.unrolled_position_first_policy, \
                    self.unrolled_position_first_value, self.unrolled_position_middle_policy,\
                        self.unrolled_position_middle_value, self.unrolled_position_last_policy,\
                            self.unrolled_position_last_value = build_network(
                                                        state=self.state_placeholder_position_network, previous_action=self.position_previous_action_placeholder, trajectory=self.trajectory,
                                                        num_action=self.num_action_position,
                                                        trajectory_state=self.trajectory_state_placeholder_position_network, trajectory_previous_action=self.trajectory_position_previous_action_placeholder
                                                        )

                self.unrolled_first_conviction_action, self.unrolled_middle_conviction_action, self.unrolled_last_conviction_action = vtrace.split_data(self.conviction_action_placeholder)
                self.unrolled_first_position_action, self.unrolled_middle_position_action, self.unrolled_position_conviction_action = vtrace.split_data(self.position_action_placeholder)
                self.unrolled_first_reward, self.unrolled_middle_reward, self.unrolled_last_reward = vtrace.split_data(self.clipped_reward_placeholder)
                self.unrolled_first_discounts, self.unrolled_middle_discounts, self.unrolled_last_discounts = vtrace.split_data(self.discounts)
                self.unrolled_first_conviction_behavior_policy, self.unrolled_middle_conviction_behavior_policy, self.unrolled_last_conviction_behavior_policy = vtrace.split_data(self.conviction_network_behaviour_policy_placeholder)
                self.unrolled_first_position_behavior_policy, self.unrolled_middle_position_behavior_policy, self.unrolled_last_position_behavior_policy = vtrace.split_data(self.position_network_behaviour_policy_placeholder)

                self.vs, self.clipped_rho = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_first_conviction_behavior_policy, target_policy_softmax=self.unrolled_conviction_first_policy,
                                                actions=self.unrolled_first_conviction_action, discounts=self.unrolled_first_discounts, rewards=self.unrolled_first_reward,
                                                values=self.unrolled_conviction_first_value, next_values=self.unrolled_conviction_middle_value, action_size=self.num_action_conviction)

                self.vs_plus_1, _ = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_middle_conviction_behavior_policy, target_policy_softmax=self.unrolled_conviction_middle_policy,
                                                actions=self.unrolled_middle_conviction_action, discounts=self.unrolled_middle_discounts, rewards=self.unrolled_middle_reward,
                                                values=self.unrolled_conviction_middle_value, next_values=self.unrolled_conviction_last_value, action_size=self.num_action_conviction)

                self.vs, self.clipped_rho = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_first_position_behavior_policy, target_policy_softmax=self.unrolled_position_first_policy,
                                                actions=self.unrolled_first_position_action, discounts=self.unrolled_first_discounts, rewards=self.unrolled_first_reward,
                                                values=self.unrolled_position_first_value, next_values=self.unrolled_position_middle_value, action_size=self.num_action_position)

                self.vs_plus_1, _ = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_middle_position_behavior_policy, target_policy_softmax=self.unrolled_position_middle_policy,
                                                actions=self.unrolled_middle_position_action, discounts=self.unrolled_middle_discounts, rewards=self.unrolled_middle_reward,
                                                values=self.unrolled_position_middle_value, next_values=self.unrolled_position_last_value, action_size=self.num_action_position)









                self.pg_advantage = tf.stop_gradient(
                    self.clipped_rho * \
                        (self.unrolled_first_reward + self.unrolled_first_discounts * self.vs_plus_1 - self.unrolled_first_value))

                self.pi_loss = vtrace.compute_policy_gradient_loss(
                    softmax=self.unrolled_first_policy,
                    actions=self.unrolled_first_action,
                    advantages=self.pg_advantage,
                    output_size=self.num_action)
                self.baseline_loss = vtrace.compute_baseline_loss(
                    vs=tf.stop_gradient(self.vs),
                    value=self.unrolled_first_value)
                self.entropy = vtrace.compute_entropy_loss(
                    softmax=self.unrolled_first_policy)

                self.total_loss = self.pi_loss + self.baseline_loss * self.baseline_loss_coef + self.entropy * self.entropy_coef

            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(self.start_learning_rate, self.num_env_frames, self.learning_frame, self.end_learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, momentum=0, epsilon=0.1)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

        self.global_to_session = copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def save_weight(self, path):
        self.saver.save(self.sess, path)

    def load_weight(self, path):
        self.saver.restore(self.sess, path)

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def train(self, state, reward, action, done, behavior_policy, previous_action, initial_h, initial_c):
        normalized_state = np.stack(state) / 255
        feed_dict={
            self.t_s_ph: normalized_state,
            self.t_pa_ph: previous_action,
            self.t_initial_h_ph: initial_h,
            self.t_initial_c_ph: initial_c,
            self.a_ph: action,
            self.d_ph: done,
            self.r_ph: reward,
            self.b_ph: behavior_policy}

        pi_loss, value_loss, entropy, learning_rate, _ = self.sess.run(
            [self.pi_loss, self.baseline_loss, self.entropy, self.learning_rate, self.train_op],
            feed_dict=feed_dict)
        
        return pi_loss, value_loss, entropy, learning_rate

    def test(self):
        batch_size = 2
        trajectory_state = np.random.rand(batch_size, self.trajectory, *self.input_shape)
        trajectory_previous_action = []
        for _ in range(batch_size):
            previous_action = [np.random.choice(self.num_action) for U in range(self.trajectory)]
            trajectory_previous_action.append(previous_action)
        trajectory_initial_h = np.random.rand(batch_size, self.trajectory, self.lstm_hidden_size)
        trajectory_initial_c = np.random.rand(batch_size, self.trajectory, self.lstm_hidden_size)

        first_value, middle_value, last_value = self.sess.run(
            [self.unrolled_first_value, self.unrolled_middle_value, self.unrolled_last_value],
            feed_dict={
                self.t_s_ph: trajectory_state,
                self.t_pa_ph: trajectory_previous_action,
                self.t_initial_h_ph: trajectory_initial_h,
                self.t_initial_c_ph: trajectory_initial_c})

        print(first_value)
        print('#####')
        print(middle_value)
        print('#####')
        print(last_value)
        print('#####')

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def get_policy_and_action(self, state, previous_action, h, c):
        normalized_state = np.stack(state) / 255
        policy, result_c, result_h = self.sess.run(
            [self.policy, self.c, self.h], feed_dict={
                                            self.s_ph: [normalized_state],
                                            self.pa_ph: [previous_action],
                                            self.initial_h_ph: [h],
                                            self.initial_c_ph: [c]})
        policy = policy[0]
        result_c = result_c[0]
        result_h = result_h[0]
        action = np.random.choice(self.num_action, p=policy)
        return action, policy, max(policy), result_c, result_h

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    
    np.random.seed(0)
    tf.set_random_seed(0)

    sess = tf.Session()
    impala = IMPALA(
                trajectory=20,
                input_shape=[84, 84, 4],
                num_action=3,
                discount_factor=0.999,
                start_learning_rate=0.0006,
                end_learning_rate=0,
                learning_frame=1000000000,
                baseline_loss_coef=0.5,
                entropy_coef=0.01,
                gradient_clip_norm=40,
                reward_clipping='abs_one',
                model_name='actor_0',
                learner_name='learner',
                lstm_hidden_size=256)
    impala.set_session(sess)
    impala.test()