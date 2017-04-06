from datasets import dataset_factory
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np



class SlimTrainMgr():
    def __init__(self):
        self.dataset_name = 'flowers'
        self.dataset_split_name = 'train'
        self.dataset_dir = '/home/levin/workspace/detection/data/flower'
        
        self.num_readers = 4
        self.batch_size = 32
        self.labels_offset = 0
        self.train_image_size = None
        self.model_name = 'inception_v3' #'The name of the architecture to train.'
        self.weight_decay = 0.00004 # 'The weight decay on the model weights.'
        
        self.preprocessing_name = None
        self.num_preprocessing_threads = 4
        self.num_epochs_per_decay = 2.0
        self.learning_rate_decay_type = 'exponential'
        self.end_learning_rate =  0.0001
        self.learning_rate = 0.01
        
        #optimiser
        self.optimizer = 'rmsprop'
        
        
        self.adadelta_rho = 0.95
        self.opt_epsilon= 1.0
        self.adagrad_initial_accumulator_value= 0.1
        self.adam_beta1= 0.9
        self.adam_beta2= 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1= 0.0
        self.ftrl_l2 = 0.0
        self.momentum= 0.9
        
        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9
        
        self.train_dir = '/tmp/tfmodel/'
        self.max_number_of_steps = None
        self.log_every_n_steps = 10
        self.save_summaries_secs = 600
        self.save_interval_secs= 600
        
        self.checkpoint_path = None
        self.checkpoint_exclude_scopes = None
        self.ignore_missing_vars = False
        
        
        
        
        self.label_smoothing = 0
        return
    def __get_images_labels(self):
        dataset = dataset_factory.get_dataset(
                self.dataset_name, self.dataset_split_name, self.dataset_dir)
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=self.num_readers,
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= self.labels_offset
        
        network_fn = nets_factory.get_network_fn(
                self.model_name,
                num_classes=(dataset.num_classes - self.labels_offset),
                weight_decay=self.weight_decay,
                is_training=True)
 
        train_image_size = self.train_image_size or network_fn.default_image_size
         
        preprocessing_name = self.preprocessing_name or self.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=True)
 
        image = image_preprocessing_fn(image, train_image_size, train_image_size)
 
        images, labels = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                num_threads=self.num_preprocessing_threads,
                capacity=5 * self.batch_size)
        labels = slim.one_hot_encoding(
                labels, dataset.num_classes - self.labels_offset)
        batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2)
        images, labels = batch_queue.dequeue()
        
        self.network_fn = network_fn
        self.dataset = dataset
        
        #set up the network
        
        return images, labels
    def __configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate.
    
        Args:
            num_samples_per_epoch: The number of samples in each epoch of training.
            global_step: The global_step tensor.
    
        Returns:
            A `Tensor` representing the learning rate.
    
        Raises:
            ValueError: if
        """
        decay_steps = int(num_samples_per_epoch / self.batch_size *
                                            self.num_epochs_per_decay)
       
    
        if self.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.learning_rate,
                                                                                global_step,
                                                                                decay_steps,
                                                                                self.learning_rate_decay_factor,
                                                                                staircase=True,
                                                                                name='exponential_decay_learning_rate')
        elif self.learning_rate_decay_type == 'fixed':
            return tf.constant(self.learning_rate, name='fixed_learning_rate')
        elif self.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.learning_rate,
                                                                             global_step,
                                                                             decay_steps,
                                                                             self.end_learning_rate,
                                                                             power=1.0,
                                                                             cycle=False,
                                                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                                         self.learning_rate_decay_type)
        return
    def __configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.
    
        Args:
            learning_rate: A scalar or `Tensor` learning rate.
    
        Returns:
            An instance of an optimizer.
    
        Raises:
            ValueError: if FLAGS.optimizer is not recognized.
        """
        if self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=self.adadelta_rho,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=self.adagrad_initial_accumulator_value)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=self.adam_beta1,
                    beta2=self.adam_beta2,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=self.ftrl_learning_rate_power,
                    initial_accumulator_value=self.ftrl_initial_accumulator_value,
                    l1_regularization_strength=self.ftrl_l1,
                    l2_regularization_strength=self.ftrl_l2)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=self.momentum,
                    name='Momentum')
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=self.rmsprop_decay,
                    momentum=self.rmsprop_momentum,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
        return optimizer
    def __get_variables_to_train(self):
        """Returns a list of variables to train.
    
        Returns:
            A list of variables to train by the optimizer.
        """
        if self.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.trainable_scopes.split(',')]
    
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train
    
    
    def __setup_training(self,images, labels):
        tf.logging.set_verbosity(tf.logging.INFO)
        logits, end_points = self.network_fn(images)

        #############################
        # Specify the loss function #
        #############################
        loss_1 = None
        if 'AuxLogits' in end_points:
            loss_1 = tf.losses.softmax_cross_entropy(
                    logits=end_points['AuxLogits'], onehot_labels=labels,
                    label_smoothing=self.label_smoothing, weights=0.4, scope='aux_loss')
        total_loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels,
                label_smoothing=self.label_smoothing, weights=1.0)
        
        if loss_1 is not None:
            total_loss = total_loss + loss_1 
        
        
        
        global_step = slim.create_global_step()
        
        # Variables to train.
        variables_to_train = self.__get_variables_to_train()
        
        learning_rate = self.__configure_learning_rate(self.dataset.num_samples, global_step)
        optimizer = self.__configure_optimizer(learning_rate)
        
        
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)
        
        self.__add_summaries(end_points, learning_rate, total_loss)
        
        ###########################
        # Kicks off the training. #
        ###########################
       
        slim.learning.train(
                train_op,
                logdir=self.train_dir,
                init_fn=self.__get_init_fn(),
                number_of_steps=self.max_number_of_steps,
                log_every_n_steps=self.log_every_n_steps,
                save_summaries_secs=self.save_summaries_secs,
                save_interval_secs=self.save_interval_secs)
        
        
        return
    def __add_summaries(self,end_points,learning_rate,total_loss):
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            tf.summary.scalar('losses/%s' % loss.op.name, loss)
        # Add total_loss to summary.
        tf.summary.scalar('total_loss', total_loss)

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        tf.summary.scalar('learning_rate', learning_rate)

        return
    def __get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training.
    
        Note that the init_fn is only run when initializing the model during the very
        first global step.
    
        Returns:
            An init function run by the supervisor.
        """
        
        if self.checkpoint_path is None:
            return None
    
        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(self.train_dir):
            tf.logging.info(
                    'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                    % self.train_dir)
            return None
    
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                                        for scope in self.checkpoint_exclude_scopes.split(',')]
    
        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
    
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
    
        tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
        return slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore,
                ignore_missing_vars=self.ignore_missing_vars)
    
    def run(self):
        
        #fine tune the new parameters
        self.train_dir = '/tmp/flowers-models/inception_v3'
        self.dataset_name = 'flowers'
        self.dataset_split_name = 'train'
        self.dataset_dir = '/home/levin/workspace/detection/data/flower'
        self.model_name = 'inception_v3'
        self.checkpoint_path = '/home/levin/workspace/detection/data/trained_models/inception_v3/inception_v3.ckpt'
        self.checkpoint_exclude_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
        self.trainable_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
        self.max_number_of_steps = 1000
        self.batch_size= 32
        self.learning_rate = 0.01
        self.learning_rate_decay_type = 'fixed'
        self.save_interval_secs = 60
        self.save_summaries_secs= 60
        self.log_every_n_steps = 100
        self.optimizer = 'rmsprop'
        self.weight_decay = 0.00004
        
        #fine tune all parameters
        self.train_dir = '/tmp/flowers-models/inception_v3/all'
        
        self.checkpoint_path = '/tmp/flowers-models/inception_v3'
        self.checkpoint_exclude_scopes = None
        self.trainable_scopes = None
        
        self.max_number_of_steps = 500
        self.learning_rate=0.0001
        self.log_every_n_steps = 10
       
        
        images, labels = self.__get_images_labels()
        self.__setup_training(images, labels)
        return
    
    


if __name__ == "__main__":   
    obj= SlimTrainMgr()
    obj.run()