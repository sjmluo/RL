import numpy as np
import random
from collections import deque
import gym

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam

class ActorCritic:
    def __init__(self, env, sess, lr=0.001, epsilon=1.0, epsilon_decay=0.995, gamma=0.95, memory_len=2000):
        """
        Parameters:
        -----------
        env: OpenAI Gym class environment.
            An environment created by the openAI Gym toolkit.
        sess: tensorflow.python.client.session.Session
            The tensor flow session.
        lr: float
            The learning rate.
        epsilon: float
            The probability of random exploration
        epsilon_decay: float
            The decay of the random exploration.
        gamma: float
            Discount factor for future reward.
        
        Returns:
        --------
        None
        """
        self.env = env
        self.sess = sess

        # Initalise Model Parameters
        self.lr = lr  # Learning rate (alpha)
        self.epsilon = epsilon  # Probability of random exploration
        self.epsilon_decay = epsilon_decay  # Decay factor for random exploration
        self.gamma = gamma  # discount factor
        
        self.memory = deque(maxlen=memory_len)

        # Initalise actor
        self.actor_state_input, self.actor_model = self._create_actor_model()
        _, self.target_actor_model = self._create_actor_model()

        # Initalise critic
        self.critic_state_input, self.critic_action_input, self.critic_model = self._create_critic_model()
        _, _, self.target_critic_model = self._create_critic_model()

        """
        Calculate de/dA = de/dC * dC/dA, where e:error, C:critic, A:actor 
        """

        # dC / dA
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

        # dC / dA
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        # de / dC
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
        self.sess.run(tf.global_variables_initializer())

    def _create_actor_model(self, N_state_h1=24,N_state_h2=48, N_state_h3=24):
        """
        Create a three layer neural network with a mapping from the state space
        to the action space.
        
        Parameters:
        -----------
        N_state_h1: int
            Number of hidden nodes in the first hidden layer.
        N_state_h2: int
            Number of hidden nodes in the second hidden layer.
        N_state_h3: int
            Number of hidden nodes in the third hidden layer.
        
        Returns:
        --------
        state_input: Keras tensor
            The state input.
        model: Keras model
            The actor model. The model is a three layer neural network.
            The neural network maps the state space to action space.
        """
        # Map state space to action space
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(N_state_h1, activation='relu')(state_input)
        state_h2 = Dense(N_state_h2, activation='relu')(state_h1)
        state_h3 = Dense(N_state_h3, activation='relu')(state_h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(state_h3)

        # Create model with mapping from state space to action space
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def _create_critic_model(self, N_state_h1=24, N_state_h2=48, N_merged_h1=24):
        """
        Create a three layer neural network with a mapping from the state space
        + action space to score.
        
        Parameters:
        -----------
        N_state_h1: int
            Number of hidden nodes in the first hidden layer.
        N_state_h2: int
            Number of hidden nodes in the second hidden layer.
            This is also the first hidden layer in the action space.
        N_merged_h1: int
            A hidden layer after the second hidden layer. This
            layer merges the state space and action space together.

        Returns:
        --------
        state_input: Keras tensor
            The state input.
        action_input: Keras tensor
            The action input.
        model: Keras model
            The actor model. The model is a three layer neural network.
            The neural network maps the state space + action space to a score.
        """
        # State input
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(N_state_h1, activation='relu')(state_input)
        state_h2 = Dense(N_state_h2)(state_h1)

        # Action input
        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(N_state_h2)(action_input)

        # Merge state and action input
        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(N_merged_h1, activation='relu')(merged)
        
        # Get score
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)
        
        # Create model with mapping from state space + action space to a score
        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        """
        Remember the previous states, actions, rewards for each episode.
        
        Parameters:
        -----------
        cur_state: numpy.ndarray
            Current state space.
        action: numpy.ndarray
            Current action.
        reward: float
            Amount of reward returned after previous action.
        new_state: numpy.ndarray
            State space after action is performed.
        done: bool
            Whether the episode has ended. If episode has ended some
            results will be undefined.

        Returns:
        --------
        None
        """
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        """
        d(theta) = alpha grad_{theta} (log pi_{theta} (s,a)) hat{q}_{w} (s,a)
        
        Parameters:
        -----------
        samples: list
            A list containing a tuple which contains information on the.
            Current state, action, reward, new state and done.

        Returns:
        --------
        None
        """
        for sample in samples:
            # Predict best action
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            
            # dC / dA
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]
            
            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        """
        d(w) = beta (R(s,a) + gamma hat{q}_{w} (s_{t+1}, a_{t+1}) - hat{q}_{w} (s_{t}, a_{t})) grad_{w} hat{q}_{w} (s_{t}, a{t})
        
        Parameters:
        -----------
        samples: list
            A list containing a tuple which contains information on the.
            Current state, action, reward, new state and done.

        Returns:
        --------
        None
        """
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                # Bellman equation / Discounting future rewards
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward  # rewards further away are worth less
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self, batch_size=32):
        """
        Train the actor and critic models. Does not run
        if the number of previous stages rememebered is
        less than the batch size.
        
        Parameters:
        -----------
        batch_size: int
            Number of training samples used to train the model.
        
        Returns:
        --------
        None
        """
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def act(self, cur_state):
        """
        Predict the action given the current state. Contains random
        chance to sample from all action space to explore.
        
        Parameters:
        -----------
        cur_state: numpy.ndarray
            The current state space
        
        Returns:
        --------
        action: numpy.ndarray
            The output action. The output action could either be random
            or the predicted action depending on epsilon.
        """
        # Decay for random exploration
        self.epsilon *= self.epsilon_decay

        # Random exploration
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.actor_model.predict(cur_state)
        return action

    def run_trials(self, trial_len=500, num_trials=1000, verbose=False):
        """
        To Do. Writing and testing this section at the moment.
        """
        for trial in range(num_trials):
            if verbose:
                print('trial:{}, epsilon:{}'.format(trial, self.epsilon))
            cur_state = self.env.reset()
            for step in range(trial_len):
                self.env.render()
                cur_state = cur_state.reshape((1, self.env.observation_space.shape[0]))
                action = self.act(cur_state)
                action = action.reshape((1, self.env.action_space.shape[0]))

                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.reshape((1, self.env.observation_space.shape[0]))

                self.remember(cur_state, action, reward, new_state, done)
                self.train()

                cur_state = new_state