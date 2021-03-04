import tensorflow as tf
import numpy as np
import model_impala
import buffer_queue_imapala



flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS



flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_integer('batch_size', 32, 'how many batch learner should be training')
flags.DEFINE_integer('queue_size', 128, 'fifoqueue size')
flags.DEFINE_integer('trajectory', 20, 'trajectory length')
flags.DEFINE_integer('learning_frame', int(1e9), 'trajectory length')
flags.DEFINE_integer('lstm_size', 256, 'lstm_size')

flags.DEFINE_float('start_learning_rate', 0.0006, 'start_learning_rate')
flags.DEFINE_float('end_learning_rate', 0, 'end_learning_rate')
flags.DEFINE_float('discount_factor', 0.99, 'discount factor')
flags.DEFINE_float('entropy_coef', 0.05, 'entropy coefficient')
flags.DEFINE_float('baseline_loss_coef', 0.5, 'baseline coefficient')
flags.DEFINE_float('gradient_clip_norm', 40.0, 'gradient clip norm')

flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'], 'Job name. Ignored when task is set to -1')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'], 'Reward clipping.')

def main(_):

    if is_single_machine():
    local_job_device = ''
    shared_job_device = ''
    is_actor_fn = lambda i: True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    filters = []
  else:
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # actors. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']
    })
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
    filters = [shared_job_device, local_job_device]

    with tf.device(shared_job_device):

        with tf.device('/cpu'):
            ''' we are saving queue on the cpu '''
            queue = buffer_queue_imapala.FIFOQueue(
                FLAGS.trajectory, input_shape_conviction , input_shape_position, output_size_conviction, output_size_position,
                FLAGS.queue_size, FLAGS.batch_size,
                FLAGS.num_actors)      
        
        learner = model_impala.IMPALA(
            trajectory=FLAGS.trajectory,
            input_shape_conviction = input_shape_conviction,
            input_shape_position = input_shape_position,
            num_action_conviction = output_size_conviction,
            num_action_position = output_size_position,
            discount_factor=FLAGS.discount_factor,
            start_learning_rate=FLAGS.start_learning_rate,
            end_learning_rate=FLAGS.end_learning_rate,
            learning_frame=FLAGS.learning_frame,
            baseline_loss_coef=FLAGS.baseline_loss_coef,
            entropy_coef=FLAGS.entropy_coef,
            gradient_clip_norm=FLAGS.gradient_clip_norm,
            reward_clipping=FLAGS.reward_clipping,
            model_name='learner',
            learner_name='learner',
            )
        
        #learner.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))
    
    with tf.device(local_job_device):
        actor = model_impala.IMPALA(
                trajectory=FLAGS.trajectory,
                input_shape_conviction = input_shape_conviction,
                input_shape_position = input_shape_position,
                num_action_conviction = output_size_conviction,
                num_action_position = output_size_position,
                discount_factor=FLAGS.discount_factor,
                start_learning_rate=FLAGS.start_learning_rate,
                end_learning_rate=FLAGS.end_learning_rate,
                learning_frame=FLAGS.learning_frame,
                baseline_loss_coef=FLAGS.baseline_loss_coef,
                entropy_coef=FLAGS.entropy_coef,
                gradient_clip_norm=FLAGS.gradient_clip_norm,
                reward_clipping=FLAGS.reward_clipping,
                model_name='actor_{}'.format(FLAGS.task),
                learner_name='learner',
                )

    sess = tf.Session(server.target)
    queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess) 
    
    if is_learner:
        '''we will write the trainer that we have here so as to train the model it will need modification backpropagation implementation and sending the returns'''
        writer = tensorboardX.SummaryWriter('runs/learner')
        train_step = 0
        
        while True:
            size = queue.get_size()
            if size > 3 * FLAGS.batch_size:
                train_step += 1
                batch = queue.sample_batch()
                s = time.time()
                pi_loss, baseline_loss, entropy, learning_rate = learner.train(
                                                                    state=np.stack(batch.state),
                                                                    reward=np.stack(batch.reward),
                                                                    action=np.stack(batch.action),
                                                                    done=np.stack(batch.done),
                                                                    behavior_policy=np.stack(batch.behavior_policy),
                                                                    previous_action=np.stack(batch.previous_action),
                                                                    initial_h=np.stack(batch.previous_h),
                                                                    initial_c=np.stack(batch.previous_c))
                writer.add_scalar('data/pi_loss', pi_loss, train_step)
                writer.add_scalar('data/baseline_loss', baseline_loss, train_step)
                writer.add_scalar('data/entropy', entropy, train_step)
                writer.add_scalar('data/learning_rate', learning_rate, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)
                print('training')
                if train_step % 1000 == 0:
                    learner.save_weight('breakout/model')
    
    else:
        trajectory_data = collections.namedtuple(
                'trajectory_data',
                ['conviction_network_state','position_network_state' ,'conviction_network_next_state','position_network_next_state' ,
                 'reward', 'done','convition_behavior_policy' ,'position_behavior_policy' , 'conviction_action', 'conviction_previous_action' , 'position_action', 'position_previous_action'])

        #env = wrappers.make_uint8_env(env_name)
        #state = env.reset()
        #previous_action = 0
        #previous_h = np.zeros([FLAGS.lstm_size])
        #previous_c = np.zeros([FLAGS.lstm_size])

        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        lives = 5

        writer = tensorboardX.SummaryWriter('runs/{}/actor_{}'.format(env_name, FLAGS.task))

        while True:

            unroll_data = trajectory_data(
                [], [], [], [], [], []
                [], [], [] ,[], [] ,[])

            actor.parameter_sync()

            for _ in range(FLAGS.trajectory):

                conviction_action, position_action, convition_behavior_policy, position_behavior_policy, max_prob = actor.get_policy_and_action(
                    conviction_network_state, position_network_state, conviction_previous_action , position_previous_action) # Need to write a function in trainer.py which takes input the state,previous action and returns action,behaviour_policy,max_prob 

                episode_step += 1
                total_max_prob += max_prob

                conviction_network_next_state, position_network_next_state , reward, done, info = env.step(action % available_output_size)

                score += reward

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                unroll_data.conviction_network_state.append(conviction_network_state)
                unroll_data.position_network_state.append(position_network_state)
                unroll_data.conviction_network_next_state.append(conviction_network_next_state)
                unroll_data.position_network_next_state.append(position_network_next_state)
                unroll_data.reward.append(r)
                unroll_data.done.append(d)
                unroll_data.conviction_action.append(conviction_action)
                unroll_data.position_action.append(position_action)
                unroll_data.convition_behavior_policy.append(convition_behavior_policy)
                unroll_data.position_behavior_policy.append(position_behavior_policy)
                unroll_data.conviction_previous_action.append(conviction_previous_action)
                unroll_data.position_previous_action.append(position_previous_action)


                conviction_network_state = conviction_network_next_state
                position_network_state = position_network_next_state

                conviction_previous_action = conviction_action
                position_previous_action = position_action
                lives = info['ale.lives']

                if done:
                    print(episode, score)
                    writer.add_scalar('data/{}/prob'.format(env_name), total_max_prob / episode_step, episode)
                    writer.add_scalar('data/{}/score'.format(env_name), score, episode)
                    writer.add_scalar('data/{}/episode_step'.format(env_name), episode_step, episode)
                    episode += 1
                    score = 0
                    episode_step = 0
                    total_max_prob = 0
                    lives = 5
                    state = env.reset()
                    previous_action = 0


            queue.append_to_queue(
                task=FLAGS.task, unrolled_state_convition_network = unroll_data.conviction_network_state,
                unrolled_state_position_network = unroll_data.position_network_state,
                unrolled_next_state_convition_network=unroll_data.conviction_network_next_state,
                unrolled_next_state_position_network=unroll_data.position_network_next_state,
                unrolled_reward=unroll_data.reward,
                unrolled_done=unroll_data.done,
                unrolled_conviction_action=unroll_data.conviction_action,
                unrolled_position_action=unroll_data.position_action,
                unrolled_conviction_behavior_policy=unroll_data.convition_behavior_policy,
                unrolled_position_behavior_policy=unroll_data.position_behavior_policy,
                unrolled_conviction_previous_action=unroll_data.convition_previous_action
                unrolled_position_previous_action=unroll_data.position_previous_action)


if __name__ == '__main__':
    tf.app.run()
