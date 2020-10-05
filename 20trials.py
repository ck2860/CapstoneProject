import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow.compat.v1 as tf
##@package TWENTYtrials
## Documentation for 20trials.py
#
# 20 trials of evaluation were performed with 20 different random seeds.Epsilon-Decreasing, Epsilon-Greedy, Hybrid#1-#5 are included in the evaluations.
adsDF= pd.read_csv('data/Ads_Optimisation.csv')

meansDF = adsDF.mean()
newarr = np.array_split(meansDF, 2)
data = np.array([newarr[0], newarr[1]])
data = np.negative([newarr[0], newarr[1]])

## Documentation for the Contextual Bandit class
# this class initializes with a state, bandits, number of bandits, and number of actions
class contextual_bandit():
    def __init__(self):
        # create a new Contextual Bandit
        self.state = 0 # The number of current bandit
        self.bandits = np.array(data)
        #self.bandits = np.random.uniform(low=-10, high=10, size=(4,4))
        self.num_bandits = self.bandits.shape[0] # the bandit number  (3)
        self.num_actions = self.bandits.shape[1] # the action number  (4)

    ## Getting the bandit aka state
    # returns a random state for every time.
    def getBandit(self):
        self.state = np.random.randint(0,len(self.bandits))
        return self.state

    ## Pulling an arm
    # return a positive or negative reward
    def pullArm(self,action):
        #Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

## Documentation for the agent class
# this class sets up a contextual bandit agent via building a feed-forward in the network and training/updating the network.
class agent():
    ## create a new agent
    # The agent would have a learning rate, the size of a state, and the size of actions.
    def __init__(self, lr, s_size,a_size):
        self.state_in= tf.placeholder(tf.int32,shape=(1,))

        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        # layer
        state_in_OH = tf.one_hot(self.state_in, s_size)
        output = tf.layers.dense(state_in_OH, a_size, tf.nn.sigmoid, use_bias=False, kernel_initializer = tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(tf.float32, shape=(1,))
        self.action_holder = tf.placeholder(tf.int32, shape=(1,))

        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


df1a = pd.DataFrame({'x': [], 'y': []})
df2a = pd.DataFrame({'x': [], 'y': []})
df3a = pd.DataFrame({'x': [], 'y': []})
df4a = pd.DataFrame({'x': [], 'y': []})
df5a = pd.DataFrame({'x': [], 'y': []})
df6a = pd.DataFrame({'x': [], 'y': []})
df7a = pd.DataFrame({'x': [], 'y': []})
a1 = 1
b = 1
c = 1
d = 1
f = 1
e1 = 1
g = 1
for a in range(20):
    ## Decreasing Epsilon
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 1 #start with highly explorative (100% explore, 0% exploit)
    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(a1)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.
            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit
            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:
                meanR = np.mean(total_reward,axis=1)
                df1a = df1a.append({'x': i, 'y': meanR[0]}, ignore_index=True)
                print("Epsilon-Decreasing mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(a1))
            e-=0.0001 #in the end it would be highly exploitative
            i+=1
        a1+=1
        print("Epsilon-Decreasing is done!")

    ## Epsilon-Greedy
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    # this is the probability of the agent to explore. 10% to explore and 90% of exploit
    e = 0.1
    init = tf.global_variables_initializer()
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(b)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.
            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit

            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.
            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:
                meanR = np.mean(total_reward,axis=1)
                df2a = df2a.append({'x': i, 'y': meanR[0]}, ignore_index=True)
                print("Epsilon-Greedy mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(b))
            i+=1
        b+=1
        print("Epsilon-Greedy is done!")

    ## Hybrid#1
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 0.9 #start with highly explorative (90% explore, 10% exploit)
    init = tf.global_variables_initializer()
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(c)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.
            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit
            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.
            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:
                meanR = np.mean(total_reward,axis=1)
                df3a = df3a.append({'x': i, 'y': meanR[0]}, ignore_index=True)
                print("Hybrid#1 mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(c))
            e-=0.00008 #in the end it would be highly exploitative (10% explore, 90% exploit)
            i+=1
        c+=1
        print("Hybrid#1 is done!")


    ## Hybrid#2
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 1 #start with highly explorative (90% explore, 10% exploit)
    init = tf.global_variables_initializer()
    #fig, ax = plt.subplots()
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(d)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.

            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit
            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:

                meanR = np.mean(total_reward,axis=1)
                df4a = df4a.append({'x': i, 'y': meanR[0]}, ignore_index=True)
                print("Hybrid#2 mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(d))
            e-=0.00018 #in the end it would be highly exploitative (10% explore, 90% exploit)
            if e < 0.1:
                e = 0.1
            i+=1
        d+=1
        print("Hybrid#2 is done!")
    ## Hybrid#3
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 0.9 #start with highly explorative (90% explore, 10% exploit)

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(e1)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.

            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit

            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:

                meanR = np.mean(total_reward,axis=1)
                df5a = df5a.append({'x': i, 'y': meanR[0],'trial': 'trial'+ str(a+1)}, ignore_index=True)
                print("Hybrid#3 mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(e1))
            e-=0.00016 #in the end it would be highly exploitative (10% explore, 90% exploit)
            if e < 0.1:
                e = 0.1
            i+=1
        e1+=1
        print("Hybrid#3 is done!")
    ## Hybrid#4
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 1 #start with highly explorative (90% explore, 10% exploit)

    init = tf.global_variables_initializer()
    ## Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(f)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.
            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit
            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:

                meanR = np.mean(total_reward,axis=1)
                df6a = df6a.append({'x': i, 'y': meanR[0],'trial': 'trial'+ str(a+1)}, ignore_index=True)
                print("Hybrid#4 mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(f))
            e-=0.00036 #in the end it would be highly exploitative (10% explore, 90% exploit)
            if e < 0.1:
                e = 0.1
            i+=1
        f+=1
        print("Hybrid#4 is done!")
    ## Hybrid#5
    tf.reset_default_graph()
    cBandit = contextual_bandit()
    myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
    e = 0.9 #start with highly explorative (90% explore, 10% exploit)

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        np.random.seed(g)
        while i < total_episodes:
            s = cBandit.getBandit() #Get a state from the environment.

            #Choose either a random action or one from our network.
            r = np.random.rand(1)
            if r < e:
                #               if np.random.rand(1) < e:
                action = np.random.randint(cBandit.num_actions) #explore
            else:
                action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) #exploit

            reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 500 == 0:

                meanR = np.mean(total_reward,axis=1)
                df7a = df7a.append({'x': i, 'y': meanR[0],'trial': 'trial'+ str(a+1)}, ignore_index=True)
                print("Hybrid#5 mean rewards: " + str(meanR[0]) + " at the episode of "+ str(i) + " in trial# " + str(g))
            e-=0.00032 #in the end it would be highly exploitative (10% explore, 90% exploit)
            if e < 0.1:
                e = 0.1
            i+=1
        g+=1
        print("Hybrid#5 is done!")

df1aa = pd.DataFrame(df1a['y'].values.reshape(20,20))
df1Ameans = df1aa.mean(0)

test1=df1a
# test1.to_csv(r'C:\Users\Condy\Desktop\test1DecreasingE1.csv', index = False, header=True)

df2aa = pd.DataFrame(df2a['y'].values.reshape(20,20))
df2Ameans = df2aa.mean(0)
test2=df2a
# test2.to_csv(r'C:\Users\Condy\Desktop\test1Egreedy1.csv', index = False, header=True)


df3aa = pd.DataFrame(df3a['y'].values.reshape(20,20))
df3Ameans = df3aa.mean(0)
test3=df3a
# test3.to_csv(r'C:\Users\Condy\Desktop\test1plan1.csv', index = False, header=True)

df4aa = pd.DataFrame(df4a['y'].values.reshape(20,20))
df4Ameans = df4aa.mean(0)
test4=df4a
# test4.to_csv(r'C:\Users\Condy\Desktop\test1plan2.csv', index = False, header=True)

df5aa = pd.DataFrame(df5a['y'].values.reshape(20,20))
df5Ameans = df5aa.mean(0)
test5=df5a
# test5.to_csv(r'C:\Users\Condy\Desktop\test1plan3.csv', index = False, header=True)

df6aa = pd.DataFrame(df6a['y'].values.reshape(20,20))
df6Ameans = df6aa.mean(0)
test6=df6a
# test6.to_csv(r'C:\Users\Condy\Desktop\test1plan4.csv', index = False, header=True)

df7aa = pd.DataFrame(df7a['y'].values.reshape(20,20))
df7Ameans = df7aa.mean(0)
test7=df7a
# test7.to_csv(r'C:\Users\Condy\Desktop\test1plan5.csv', index = False, header=True)

df=pd.DataFrame({'x': df1a['x'], 'Decreasing Epsilon': df1Ameans , 'Greedy Epsilon': df2Ameans, 'Hybrid #1': df3Ameans, 'Hybrid #2': df4Ameans , 'Hybrid#3': df5Ameans, 'Hybrid#4': df6Ameans, 'Hybrid#5':df7Ameans})

# Initialize the figure
plt.style.use('seaborn-darkgrid')

# my_dpi=96
plt.figure(figsize=(900/96, 900/96), dpi=96)
# create a color palet6e
palette = plt.get_cmap('tab10')

# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1

    # Find the right spot on the plot
    plt.subplot(3,3, num)

    # plot every groups, but discreet
    for v in df.drop('x', axis=1):
        plt.plot(df['x'], df[v], marker='', color='black', linewidth=0.5, alpha=0.5)

    # Plot the lineplot
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)

    # Same limits for everybody!
    plt.xlim(0,10000)
    plt.ylim(-8,200)

    # Not ticks everywhere
    if num in range(5) :
        plt.tick_params(labelbottom=False)
    if num in [2,3,5,6] :
        plt.tick_params(labelleft=False)

        #Adding labels
    if num in range(1,8,3):
        plt.ylabel("Mean Reward",fontsize=10)
    if num in [5,6,7] :
        plt.xlabel("Episode",fontsize=10)

    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

# general title
plt.suptitle("In comparison of mean rewards", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
plt.show()
print("The Python script is %s" % (sys.argv[0]))
print("20 trials of greedy-based strategies were performed.")
