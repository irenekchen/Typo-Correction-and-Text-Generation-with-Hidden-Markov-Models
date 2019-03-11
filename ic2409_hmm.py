import sys, math, random

class HMM(object):
    # HMM model parameters #
    """
    Assume |X| state values and |E| emission values.
    initial: Initial belief probability distribution, list of size |X|
    tprob: Transition probabilities, size |X| list of size |X| lists;
           tprob[i][j] returns P(j|i), where i and j are states
    eprob: Emission probabilities, size |X| list of size |E| lists;
           eprob[i][j] returns P(j|i), where i is a state and j is an emission
    """
    def __init__(self, initial, tprob, eprob):
        self.initial = initial # initial belief distribution as a list P(x_1=i) = initial[i]
        self.tprob = tprob # tprob[i][j] : i = X_t : j = X_t+1
        self.eprob = eprob # eprob[i][k] : i = X_t : k = E_t
        # primarily indexed by current state

        """
        Model format:
        - all state values
        - inital believe distribution
        - transition probabilties matrix
        - all possible emissions values
        - emission probailities matrix

        Each item deliminated by "\n\n"
        Contents within each item, deliminated by commas CSV?
        """

        self.state_values = dict() # (timestep, state_index) : probability
        # must derive the probability value of a state_index for each timestep and use that



    # Normalize a probability distribution
    def normalize(self, pdist):
        s = sum(pdist)
        for i in range(0,len(pdist)):
            pdist[i] = pdist[i] / s
        return pdist


    # Propagation (elapgggggggse time)
    """
    Input: Current belief distribution in the hidden state P(X_{t-1}))
    Output: Updated belief distribution in the hidden state P(X_t)
    """
    def propagate(self, belief):
        update_belief = []
        for state_index in range(len(belief)):
            new_belief = []
            for trans_index in range(len(self.tprob)):
                p = self.tprob[trans_index][state_index] * belief[trans_index]
                new_belief.append(p)
            update_belief.append(math.fsum(new_belief))
        return update_belief

        

    # Observation (weight by evidence)
    """
    Input: Current belief distribution in the hidden state P(X_t),
           index corresponding to an observation e_t
    Output: Updated belief distribution in the hidden state P(X_t | e_t)  
    """
    def observe(self, belief, obs):
        update_belief = []
        for state_index in range(len(belief)):
            ems_val = self.eprob[state_index][obs]
            new_belief = ems_val * belief[state_index]
            update_belief.append(new_belief)
        total = math.fsum(update_belief)
        for index, val in enumerate(update_belief):
            update_belief[index] = val/total
        return update_belief

    # Filtering
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: Posterior belief distribution of the hidden state P(X_t | e_1, ..., e_t)
    """
    def filter(self, observations):
        observe_count = 1

        while observe_count < len(observations):
            if observe_count == 1:
                curr_belief = self.initial
            for state_index in range(len(curr_belief)):
                update_belief = self.propagate(curr_belief)
                update_belief = self.observe(update_belief, observations[observe_count])
            curr_belief = update_belief
            observe_count += 1
        return update_belief

    def viterbi(self, observations):
        observations = [None] + observations
        seq = []
        state_values = dict() 
        for timestep in range(len(observations)):
            if timestep == 0:
                current_state_distribution = self.initial 
                for curr_state_index, state_val in enumerate(self.initial):
                    state_values[(timestep, curr_state_index)] = (state_val, None)
                belief = []
                for next_state_index in range(len(self.initial)):
                    #obs = self.eprob[next_state_index][observations[timestep+1]]
                    curr_distribution = state_values[(timestep, next_state_index)][0]
                    belief.append(curr_distribution)
                update_belief = self.observe(belief, observations[timestep+1])
                for next_state_index in range(len(self.initial)):
                    state_values[(timestep+1, next_state_index)] = (update_belief[next_state_index], next_state_index)
            elif not timestep == len(observations)-1:
                max_states = []
                belief = []
                for next_state_index in range(len(self.initial)):

                    curr_distributions = []
                    for curr_state_index in range(len(self.initial)):
                        curr_distribution = self.tprob[curr_state_index][next_state_index] * state_values[(timestep, curr_state_index)][0]
                        curr_distributions.append(curr_distribution)

                    max_curr_distribution = float("-inf")
                    max_state = -1
                    for index, val in enumerate(curr_distributions):
                        if val > max_curr_distribution:
                            max_curr_distribution = val
                            max_state = index
                    belief.append(max_curr_distribution)
                    max_states.append(max_state)
                update_belief = self.observe(belief, observations[timestep+1])
                for i in range(len(max_states)):
                    state_values[timestep+1,i] = (update_belief[i], max_states[i])

        timestep = len(observations) - 1
        curr_max_state = -1
        next_max_state = -1
        max_distribution = float("-inf")
        for state_index in range(len(self.initial)):
            state_value = state_values[(timestep, state_index)]
            if state_value[0] > max_distribution:
                curr_max_state = state_index
                max_distribution = state_value[0]
                next_max_state = state_value[1]
        seq.append(curr_max_state)
        timestep -= 1
        while timestep > 0:
            seq.append(next_max_state)
            curr_max_state = next_max_state
            next_max_state = state_values[(timestep, curr_max_state)][1]
            timestep -= 1
        return reversed(seq)

    """
    def viterbi(self, observations):
        seq = []
        state_values = dict() # (timestep, state_index) : probability
        for timestep, observation in enumerate(observations):
            if timestep == 0:
                current_state_distribution = self.initial 
                #initalize dictionary
                for curr_state_index, state_val in enumerate(self.initial):
                    state_values[(timestep, curr_state_index)] = state_val
                #calc next_state_distribution
                for next_state_index in range(len(self.initial)):
                    obs = self.eprob[next_state_index][observations[timestep+1]]
                    curr_distribution = state_values[(timestep, next_state_index)]
                    #next_distribution = obs * curr_distribution
                    print(obs, curr_distribution)
                    state_values[(timestep+1, next_state_index)] = (obs * curr_distribution, next_state_index)
            elif not timestep == len(observations)-1:
                for next_state_index in range(len(self.initial)):

                    curr_distributions = []
                    # calculate max and ems
                    for curr_state_index in range(len(self.initial)):
                        curr_distribution = self.tprob[curr_state_index][next_state_index] * state_values[(timestep, curr_state_index)][0]
                        curr_distributions.append(curr_distribution)

                    max_curr_distribution = float("-inf")
                    max_state = -1
                    for index, val in enumerate(curr_distributions):
                        if val > max_curr_distribution:
                            max_curr_distribution = val
                            max_state = index

                    obs = self.eprob[next_state_index][observations[timestep+1]]
                    state_values[(timestep+1, next_state_index)] = (obs * max_curr_distribution, max_state)
        #for timestep in range(len(observations)):

        #WHEN CHECKING MAX, MAKE SURE THAT IF THE RESULT EXISTS IN DICT THAT ITS GREATER

        timestep = len(observations) - 1
        max_state = -1
        max_distribution = float("-inf")
        for state_index in range(len(self.initial)):
            state_value = state_values[(timestep, state_index)]
            print(state_value[0], max_distribution)
            if state_value[0] > max_distribution:
                max_distribution = state_value[0]
                max_state = state_value[1]
        seq.append(max_state)
        timestep -= 1
        while timestep > 0:
            next_state = state_values[(timestep, max_state)][1]
            seq.append(next_state)
            timestep -= 1

        return seq
    """


    """
    def viterbi(self, observations):
        seq = []
        state_values = dict() # (timestep, state_index) : probability
        for timestep, observation in enumerate(observations):
            if timestep == 0:
                current_state_distribution = self.initial 
                #initalize dictionary
                for curr_state_index, state_val in enumerate(self.initial):
                    state_values[(timestep, curr_state_index)] = state_val
                #calc next_state_distribution
                for next_state_index in range(len(self.initial)):
                    obs = self.eprob[next_state_index][observations[timestep+1]]
                    curr_distribution = state_values[(timestep, next_state_index)]
                    #next_distribution = obs * curr_distribution
                    state_values[(timestep+1, next_state_index)] = obs * curr_distribution
            else:
                for next_state_index in range(len(self.initial)):

                    curr_distributions = []
                    # calculate max and ems
                    for curr_state_index in range(len(self.initial)):
                        curr_distribution = self.tprob[curr_state_index][next_state_index] * state_values[(timestep, curr_state_index)]
                        curr_distributions.append(curr_distribution)
                    max_curr_distribution = max(curr_distributions)
                    obs = self.eprob[next_state_index][observations[timestep+1]]
                    state_values[(timestep+1, next_state_index)] = obs * max_curr_distribution
        #for timestep in range(len(observations)):


        timestep = len(observations) - 1
        while timestep > 0:
            max_state = -1
            max_distribution = float("-inf")
            for key, val in state_values.items():
                if val > max_distribution:
                    max_distribution = val
                    max_state = key[0]


            timestep -= 1





                    b = max(curr_distributions)
                    
                state_values[(timestep+1, next_state_index)]
    """
            #if timestep == 1:

                
   
    # Viterbi algorithm
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: List of most likely sequence of state indices [X_1, ..., X_t]
    """

    """
    def viterbi(self, observations):
        seq = []
        # self.state_values[()] # (timestep, state_index) : probability
        timestep = 1

        for timestep in range(1, len(observations)):
            for timestep, observation in enumerate(observations):
                max_probability = float("-inf")
                for state_index in range(len(self.initial)):

                    self.eprob[state_index][observation]

            for state_index in range(len(self.initial)):
                for observation in self.eprob[state_index][observations[]]:

                prev_timestep = timestep - 1
                for state_index_old in range(len(self.initial)):
                    state_value = self.eprob[state_index_old][]
                    
                    self.state_values[(timestep, state_index)] = max( of prev tiemstep)

        while timestep < len(observations):
            if observe_count == 1:
                curr_belief = self.initial

            for state_index in range(len(curr_belief)):
                max_belief = self.observe(update_belief, observations[observe_count])
                state_values[(timestep, state_index)] = max_belief
        seq.append(self.viterbi(observations[:-1]))
        # base case
        if timestep == 0:
            return self.initial


        return seq

    def viterbi(self, observations, seq):
        # get max

        for state_index in range(len(self.initial)):
            seq[state_index][1] = self.initial


        for timestep in range(1, len(observations)):
            for state_index in range(len(self.initial)):
                state_values[(timestep, state_index)] = max(state_values[(timestep-1, state_index)])
"""


# Functions for testing
# You should not change any of these functions
def load_model(filename):
    model = {}
    input = open(filename, 'r')
    i = input.readline()
    x = i.split()
    model['states'] = x[0].split(",")
    
    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['initial'] = [float(i) for i in y]

    input.readline()
    tprob = []
    for i in range(len(model['states'])):
        t = input.readline()
        x = t.split()
        y = x[0].split(",")
        tprob.append([float(i) for i in y])
    model['tprob'] = tprob

    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['emissions'] = dict(zip(y, range(len(y))))

    input.readline()
    eprob = []
    for i in range(len(model['states'])):
        e = input.readline()
        x = e.split()
        y = x[0].split(",")
        eprob.append([float(i) for i in y])
    model['eprob'] = eprob

    return model

def load_data(filename):
    input = open(filename, 'r')
    data = []
    for i in input.readlines():
        x = i.split()
        if x == [',']:
            y = [' ', ' ']
        else:
            y = x[0].split(",")
        data.append(y)
    observations = []
    classes = []
    for c, o in data:
        observations.append(o)
        classes.append(c)

    data = {'observations': observations, 'classes': classes}
    return data

def generate_model(filename, states, emissions, initial, tprob, eprob):
    f = open(filename,"w+")
    for i in range(len(states)):
        if i == len(states)-1:
            f.write(states[i]+'\n')
        else:
            f.write(states[i]+',')
    f.write('\n')

    for i in range(len(initial)):
        if i == len(initial)-1:
            f.write('%f\n'%initial[i])
        else:
            f.write('%f,'%initial[i])
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(states)):
            if j == len(states)-1:
                f.write('%f\n'%tprob[i][j])
            else:
                f.write('%f,'%tprob[i][j])
    f.write('\n')

    for i in range(len(emissions)):
        if i == len(emissions)-1:
            f.write(emissions[i]+'\n')
        else:
            f.write(emissions[i]+',')
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(emissions)):
            if j == len(emissions)-1:
                f.write('%f\n'%eprob[i][j])
            else:
                f.write('%f,'%eprob[i][j])
    f.close()


def accuracy(a,b):
    total = float(max(len(a),len(b)))
    c = 0
    for i in range(min(len(a),len(b))):
        if a[i] == b[i]:
            c = c + 1
    return c/total

def test_filtering(hmm, observations, index_to_state, emission_to_index):
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]

    print('Short observation sequence:')
    print('   ', obs_short)
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_filter = hmm.filter(obs_indices_short)
    result_filter_full = hmm.filter(obs_indices)

    print('\nFiltering - distribution over most recent state given short data set:')
    for i in range(0, len(result_filter)):
        print('   ', index_to_state[i], '%1.3f' % result_filter[i])

    print('\nFiltering - distribution over most recent state given full data set:')
    for i in range(0, len(result_filter_full)):
        print('   ', index_to_state[i], '%1.3f' % result_filter_full[i])

def test_viterbi(hmm, observations, classes, index_to_state, emission_to_index):
    n_obs_short = 10

    obs_short = observations[0:n_obs_short]
    classes_short = classes[0:n_obs_short]
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_viterbi = hmm.viterbi(obs_indices_short)
    best_sequence = [index_to_state[i] for i in result_viterbi]
    result_viterbi_full = hmm.viterbi(obs_indices)
    best_sequence_full = [index_to_state[i] for i in result_viterbi_full]

    print('\nViterbi - predicted state sequence:\n   ', best_sequence)
    print('Viterbi - actual state sequence:\n   ', classes_short)
    print('The accuracy of your viterbi classifier on the short data set is', accuracy(classes_short, best_sequence))
    print('The accuracy of your viterbi classifier on the entire data set is', accuracy(classes, best_sequence_full))


# Train a new typo correction model on a set of training data (extra credit)
"""
Input: List of t observations in the form of string or other data type literals
Output: Dictionary of HMM quantities, including 'states', 'emissions', 'initial', 'tprob', and 'eprob' 
"""
def train(observations, classes):
    return {'states': [], 'emissions': [], 'initial': [], 'tprob': [], 'eprob': []}

if __name__ == '__main__':
    # this if clause for extra credit training only
    if len(sys.argv) == 4 and sys.argv[1] == '-t':
        input = open(sys.argv[3], 'r')
        data = []
        for i in input.readlines():
            x = i.split()
            if x == [',']:
                y = [' ', ' ']
            else:
                y = x[0].split(",")
            data.append(y)

        observations = []
        classes = []
        for c, o in data:
            observations.append(o)
            classes.append(c)

        model = train(observations, classes)
        generate_model(sys.argv[2], model['states'], model['emissions'], model['initial'], model['tprob'], model['eprob'])
        exit(0)

    # main part of the assignment
    if len(sys.argv) != 3:
        print("\nusage: ./hmm.py [model file] [data file]")
        exit(0)

    model = load_model(sys.argv[1])
    data = load_data(sys.argv[2])

    new_hmm = HMM(model['initial'], model['tprob'], model['eprob'])
    test_filtering(new_hmm, data['observations'], model['states'], model['emissions'])
    test_viterbi(new_hmm, data['observations'], data['classes'], model['states'], model['emissions'])