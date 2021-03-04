import tensorflow as tf
import collections

class FIFOQueue:
    def __init__(self, trajectory, input_shape_conviction , input_shape_position, output_size_conviction, output_size_position,
                 queue_size, batch_size, num_actors): #trajectory is a number i.e.20 , input shape for us 240 * 21 and 240 * 7,output size 2 
        
        self.trajectory = trajectory
        self.input_shape_conviction = input_shape_conviction
        self.input_shape_position = input_shape_position
        self.output_size_conviction = output_size_conviction
        self.output_size_position = output_size_position
        self.batch_size = batch_size
        
        self.unrolled_state_conviction_network = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape_conviction])
        self.unrolled_state_position_network = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape_position])
        self.unrolled_next_state_conviction_network = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape_convition])
        self.unrolled_next_state_position_network = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape_position])
        self.unrolled_reward = tf.placeholder(tf.float32, [self.trajectory])
        self.unrolled_done = tf.placeholder(tf.bool, [self.trajectory])
        self.unrolled_conviction_behavior_policy = tf.placeholder(tf.float32, [self.trajectory, self.output_size_conviction])
        self.unrolled_position_behavior_policy = tf.placeholder(tf.float32, [self.trajectory, self.output_size_position])
        self.unrolled_conviction_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_position_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_conviction_previous_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_position_previous_action = tf.placeholder(tf.int32, [self.trajectory])
        
        self.queue = tf.FIFOQueue(
            queue_size,
            [self.unrolled_state_convition_network.dtype,
            self.unrolled_state_position_network.dtype,
            self.unrolled_next_state_convition_network.dtype,
            self.unrolled_next_state_position_network.dtype,
            self.unrolled_reward.dtype,
            self.unrolled_done.dtype,
            self.unrolled_conviction_behavior_policy.dtype,
            self.unrolled_position_behavior_policy.dtype,
            self.unrolled_conviction_action.dtype,
            self.unrolled_conviction_previous_action.dtype,
            self.unrolled_position_action.dtype,
            self.unrolled_position_previous_action.dtype,
            ], shared_name='buffer')

        self.queue_size = self.queue.size()
        
        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                self.queue.enqueue(
                    [self.unrolled_state_convition_network,
                     self.unrolled_state_position_network,
                     self.unrolled_next_state_convition_network,
                     self.unrolled_next_state_position_network,
                     self.unrolled_reward,
                     self.unrolled_done,
                     self.unrolled_conviction_action,
                     self.unrolled_position_action
                     self.unrolled_conviction_previous_action,
                     self.unrolled_position_previous_action,
                     self.unrolled_conviction_behavior_policy
                     self.unrolled_position_behavior_policy]))

        self.dequeue = self.queue.dequeue()

    def append_to_queue(self, task, unrolled_state_convition_network,unrolled_state_position_network, unrolled_next_state_convition_network,unrolled_next_state_position_network,
                        unrolled_reward, unrolled_done, unrolled_conviction_behavior_policy,unrolled_position_behavior_policy,
                        unrolled_conviction_action, unrolled_position_action ,  unrolled_conviction_previous_action ,unrolled_position_previous_action ):

        self.sess.run(
            self.enqueue_ops[task],
            feed_dict={
                self.unrolled_state_convition_network: unrolled_state_convition_network,
                self.unrolled_state_position_network: unrolled_state_position_network,
                self.unrolled_next_state_convition_network: unrolled_next_state_convition_network,
                self.unrolled_next_state_position_network: unrolled_next_state_position_network,
                self.unrolled_reward: unrolled_reward,
                self.unrolled_done: unrolled_done,
                self.unrolled_conviction_action: unrolled_conviction_action,
                self.unrolled_position_action: unrolled_position_action,
                self.unrolled_conviction_previous_action = unrolled_conviction_previous_action,
                self.unrolled_position_previous_action = unrolled_position_previous_action,
                self.unrolled_conviction_behavior_policy: unrolled_conviction_behavior_policy,
                self.unrolled_position_behavior_policy: unrolled_position_behavior_policy
                })

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
        ['conviction_network_state','position_network_state' ,'conviction_network_next_state','position_network_next_state' , 'reward', 'done',
         'convition_behavior_policy' ,'position_behavior_policy' , 'conviction_action', 'conviction_previous_action' , 'position_action', 'position_previous_action'])

        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        unroll_data = batch_tuple(
            [i[0] for i in batch],
            [i[1] for i in batch],
            [i[2] for i in batch],
            [i[3] for i in batch],
            [i[4] for i in batch],
            [i[5] for i in batch],
            [i[6] for i in batch],
            [i[7] for i in batch],
            [i[8] for i in batch],
            [i[9] for i in batch],
            [i[10] for i in batch],
            [i[11] for i in batch])

        return unroll_data

    def get_size(self):
        size = self.sess.run(self.queue_size)
        return size

    def set_session(self, sess):
        self.sess = sess

if __name__ == '__main__':
    
    queue = FIFOQueue(
        20, [240, 21], [240 , 7] , 2, 10 , 128, 32 )

    print(queue.unrolled_state)