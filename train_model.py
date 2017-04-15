from datasets import dataset_factory
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

import numpy as np
from preparedata import PrepareData
from nets.ssd import g_ssd_model


class TrainModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        
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
    
    
    def __start_training(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        
        #get batched training training data 
        image, gclasses, glocalisations, gscores = self.get_voc_2007_train_data()
        
        #get model outputs
        predictions, localisations, logits, end_points = g_ssd_model.get_model(image)
        
        #get model training losss
        total_loss = g_ssd_model.get_losses(logits, localisations, gclasses, glocalisations, gscores)

        
        
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
        # Add summaries for end_points (activations).

        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x))
        # Add summaries for losses and extra losses.
        
        tf.summary.scalar('total_loss', total_loss)
        for loss in tf.get_collection('EXTRA_LOSSES'):
            tf.summary.scalar(loss.op.name, loss)

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

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
        self.train_dir = './logs'
        
        
        self.checkpoint_path = '../data/trained_models/vgg16/vgg_16.ckpt'
        self.checkpoint_exclude_scopes = 'ssd_300_vgg/additional_blocks'
        self.trainable_scopes = 'ssd_300_vgg/additional_blocks'
        self.max_number_of_steps = 1000
        self.batch_size= 32
        self.learning_rate = 0.001
        self.learning_rate_decay_type = 'fixed'
        self.save_interval_secs = 600
        self.save_summaries_secs= 60
        self.log_every_n_steps = 100
        self.optimizer = 'adam'
        self.weight_decay = 0.00005
        
        #fine tune all parameters
#         self.train_dir = '/tmp/flowers-models/inception_v3/all'
#         
#         self.checkpoint_path = '/tmp/flowers-models/inception_v3'
#         self.checkpoint_exclude_scopes = None
#         self.trainable_scopes = None
#         
#         self.max_number_of_steps = 500
#         self.learning_rate=0.0001
#         self.log_every_n_steps = 10
       
        
        
        self.__start_training()
        return
    
    


if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()