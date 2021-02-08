class N_gram:
    def __init__(self, corpora_path_list, N, regexp_tokenizer=True, add_periods=False, case_sensitive=False):
        self.N = N
        self.corpora_path_list = corpora_path_list
        self.text_list = None
        self.adj_counters = None
        self.phrases_sets = None
        self.V = None
        self.regexp_tokenizer = regexp_tokenizer
        self.add_periods = add_periods
        self.case_sensitive = case_sensitive

    def train(self):
        print("Preprocessing training corpora...")
        if self.text_list is None:
            self.text_list = self.preprocess_corpus(self.corpora_path_list)
            print("Training set size:", len(self.text_list), "words")
        if self.case_sensitive:
            print("Recording all variations of capitalization...")
            self.capsVariations = self.recordCapsVariations()
        print("Constructing Phrase/Adjacency counters...")
        self.adj_counters = self.calc_adj_counters(self.N)
        print("Constructing List of Phrase Sets...")
        self.phrases_sets = [set(self.list_phrases(i)) for i in range(self.N + 1)]
        self.V = len(set(self.text_list))  # size of vocab (# unique words in corpora)
        print("Done Training.")

    def preprocess_corpus(self, text_file_path_list):
        """
        Input: String of text_file_path.
        Returns: List format of text, as divided by regex delimiters.
        Either tokenizes punctuation or not, depending on RegexpTokenizer parameter;
        Lowercases all words.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.preprocess_corpus(["carter_concession.txt"])
        ["i", "promised", "you", "four", "years", "ago", ...]
        """
        assert isinstance(text_file_path_list, list), "text_file_path_list not a list."

        def txt_to_str(text_file_path_list):
            aggregated_str = ""
            for text_file_path in text_file_path_list:
                f = open(text_file_path)
                aggregated_str += f.read()
            return aggregated_str

        raw_text = txt_to_str(text_file_path_list)
        if self.add_periods:
            sentences_list = raw_text.split("\n")
            text_list = ["."] + [token
                                 for sentence in [sentence.split() + ["."] for sentence in sentences_list]
                                 for token in sentence]
        else:
            tokenizer = nltk.tokenize.RegexpTokenizer('[\w\']+|[.$%!?]+')
            if not self.regexp_tokenizer:  # Use whitespace tokenizer
                tokenizer = nltk.tokenize.regexp.WhitespaceTokenizer()
            # Include words/numbers as one token, periods/$/% as another.
            # [\w\'] allows us to ensure that contractions like I'm are preserved as one token.
            text_list = tokenizer.tokenize(raw_text)
            if not self.case_sensitive:
                text_list = ["."] + [word.lower() for word in text_list]  # lowercase everything
        return text_list

    def list_phrases(self, n):
        """
        Creates a list of tuples of adjacent word pairs.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = ["This", "is", "Sam", "."]
        >>> c.list_phrases(2)
        [("This", "is"), ("is", "Sam"), ("Sam", ".")]
        """
        return [tuple([self.text_list[i + j] for j in range(n)])
                for i in range(0, len(self.text_list) - n + 1)]

    def normalize_dict(self, d):
        """
        Creates copy of d;
        Sums all keys of the dictionary d and divides each key by the total;
        returns the new dict.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> d = {1 : 2, 2 : 3}
        >>> c.normalize_dict(d)
        {1 : 0.4, 2 : 0.6}
        """
        normalized_dict = dict(d)
        dict_vals_total = sum(d.values())
        normalized_dict.update((key, val / dict_vals_total) for key, val in normalized_dict.items())
        return normalized_dict

    def sample_dict(self, d):
        """
        Sample a key from a dictionary of counts.
        """
        assert d, "dictionary is empty"
        d = self.normalize_dict(d)
        # print(d)
        rand = random.random()

        cumProb = 0.0
        for currKey in d:
            cumProb += d[currKey]
            if rand <= cumProb:
                return currKey
        return currKey

    def count_words(self):
        """
        Returns a dictionary of the words as keys and their counts as values.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = ["i", "need", "i", "have"]
        >>> c.count_words()
        {"i": 2, "need": 1, "have": 1}
        """
        ### START YOUR CODE HERE ###
        temp = {}
        for i in self.text_list:
            if i in temp.keys():
                temp[i] += 1
            else:
                temp[i] = 1
        return temp
        ### END YOUR CODE HERE ###

    def calc_word_probs(self):
        """
        Returns a dictionary with the sample probability of drawing each word.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = ["i", "need", "i", "have"]
        >>> c.calc_word_probs()
        {"i": 0.5, "need": 0.25, "have": 0.25}
        """
        ### START YOUR CODE HERE ###
        temp = self.count_words()
        for i in temp.keys():
            temp[i] /= len(self.text_list)
        return temp
        ### END YOUR CODE HERE ###

    def probs_to_neg_log_probs(self, probs_dict):
        """
        Convert dictionary of probabilities into dictionary of negative ln probabilities.
        """
        ### START YOUR CODE HERE ###
        neg_log_probs = self.calc_word_probs()
        for i in neg_log_probs.keys():
            neg_log_probs[i] = -np.log(neg_log_probs[i])
        ### END YOUR CODE HERE ###
        return neg_log_probs

    def calc_neg_log_word_probs(self):
        """
        Convert text list into dictionary of negative ln probabilities.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = ["i", "need", "i", "have"]
        >>> c.calc_neg_log_word_probs()
        {"i": -0.69, "need": -1.39, "have": -1.39}
        """
        word_probs = calc_word_probs()
        return self.probs_to_neg_log_probs(word_probs)

    def calc_adj_counter(self, n):
        """
        Convert text list into dictionary of counts of phrases of length n.

        >>> c = N_gram([], 2)
        >>> c.text_list = ["i", "have", "a", "dream", ".", "i", "have"]
        >>> c.calc_adj_counter(3)
        {("i", "have"): 2, ("have", "a"): 1, ("a", "dream"): 1, ("dream", "."): 1, (".", "i"): 1}
        """
        return collections.Counter(self.list_phrases(n))

    def calc_adj_counters(self, n):
        """
        Create a list of adj_counters of phrase length 1,...,n on text_list.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = ["eye", "for", "eye"]
        >>> c.calc_adj_counters(3)
        [0,
        {("eye",): 2, ("for"): 1},
        {("eye", "for"): 1, ("for", "eye"): 1}
        {("eye", "for", "eye"): 1}]
        """
        adj_counters = [0 for i in range(n + 1)]
        for i in range(1, n + 1):
            adj_counters[i] = self.calc_adj_counter(i)
        return adj_counters

    def calc_adj_probs(self, adj_counter):
        """
        Convert adj_counter values from counts to sample probabilities.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> adj_counter = {("hey", "you"): 2. ("you", "."): 3}
        >>> c.calc_adj_probs(adj_counter)
        {("hey", "you"): 0.4. ("you", "."): 0.6}
        """
        return self.normalize_dict(adj_counter)

    def filter_adj_counter(self, adj_counter, word_tuple, n):
        """
        Returns: Dictionary with num_occurrences of word pairs starting with 'word_tuple'.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> word_tuple = ("Jimmy",)
        >>> adj_counter = {("never", "lie"): 3, ("hey", "Jimmy"): 2,
            ("Jimmy", "Li"): 1, ("Jimmy", "Carter"): 100}
        >>> c.filter_adj_counter(adj_counter, word_tuple, 2)
        {("Jimmy", "Li"): 1, ("Jimmy", "Carter"): 100}
        """
        assert len(word_tuple) in range(n + 1), \
            "word_tuple does not have valid length: " + str(len(word_tuple))

        if not adj_counter:
            adj_counter = self.calc_adj_counter(n)
        subset_word_adj_counter = {}

        if not word_tuple:
            return dict(self.calc_adj_counter(1))
        elif len(word_tuple) == n:
            return {word_tuple: adj_counter[word_tuple]}

        for phrase in adj_counter.keys():
            ### START YOUR CODE HERE ###
            if phrase[:-1] == word_tuple[:]:
                subset_word_adj_counter[phrase] = adj_counter[phrase]
            ### END YOUR CODE HERE ###
        return subset_word_adj_counter

    def perplexity(self, sentence_list, n):
        """
        Returns perplexity of sentence_list given:
        --text_list: list of tokens to train on.
        --adj_counters: list of all 1,...,n adj_counter. Uses the output of calc_adj_counters(.,.)
        --n: The N in N-gram.

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.train()
        >>> sentence_list = ["super", "confusing", "sentence"]
        >>> c.perplexity(sentence_list, 2)
        2000.01
        """
        text_neg_log_prob = self.calc_neg_log_prob_of_sentence(sentence_list, n, self.p_KN)
        return np.exp(text_neg_log_prob / len(sentence_list))

    def p_naive(self, curr_word, prev_phrase, n):
        """
        Calculates the sample probability of all length-`n` phrases in `text_list`

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.text_list = [".", "i", "think", ".", "i", "am", ".", "why", "yes", "."]
        >>> c.p_naive("i", (".",), 2)
        0.666 # Because 3 length-2 phrases start with (".",), and of the three, 2 of them end with "curr_word"
        """
        assert n >= 1, "Invalid value of n."
        assert len(prev_phrase) < n, "Length of prev_phrase not < n."
        assert isinstance(prev_phrase, tuple), "prev_phrase is not a tuple: " + str(prev_phrase)
        filtered_adj_counter = self.filter_adj_counter(None, prev_phrase, n)
        try:
            ### START YOUR CODE HERE ###
            temp = self.normalize_dict(filtered_adj_counter)
            for i in temp.keys():
                if i[-1] == curr_word:
                    prob = temp[i]
            ### END YOUR CODE HERE ###
        except KeyError:
            print(prev_phrase + (curr_word,), " has probability 0.")
            prob = 0
        return prob

    def p_laplace(self, curr_word, prev_phrase, n, k=0.1):
        """
        Calculates the sample probability of all length-`n` phrases in `text_list`
        And performs k-smoothing.
        """
        assert n >= 1, "Invalid value of n."
        assert len(prev_phrase) < n, "Length of prev_phrase not < n."
        assert isinstance(prev_phrase, tuple), "prev_phrase is not a tuple: " + str(prev_phrase)

        filtered_adj_counter = self.filter_adj_counter(None, prev_phrase, n)
        num_entries = len(filtered_adj_counter.keys())
        laplace_denominator = sum(filtered_adj_counter.values()) + k * (num_entries + 1)
        smoothed_prob_dict = dict(filtered_adj_counter)
        smoothed_prob_dict.update((key, (val + k) / laplace_denominator) for key, val in smoothed_prob_dict.items())

        unk_prob = k / laplace_denominator  # out of distribution words

        err_tolerance = 1e-4
        err_msg = "Smoothed probabilities don't add up to 1!"

        assert np.abs(sum(smoothed_prob_dict.values()) + unk_prob - 1) < err_tolerance, err_msg

        try:
            prob = smoothed_prob_dict[prev_phrase + (curr_word,)]
        except KeyError:
            prob = unk_prob
            # print(prev_phrase + (curr_word,), " has probability 0. Replaced with prob", prob)
        # print("smoothed_prob_dict", smoothed_prob_dict)
        # print("prev_phrase", prev_phrase)
        # print("curr_word", curr_word)
        return prob

    def calc_neg_log_prob_of_sentence(self, sentence_list, n, p_func=p_laplace):
        """
        Return negative log probability of `sentence_list` occurring, given:
        --n: The n in n-gram.
        --p_func: must contain args (curr_word, prev_phrase, text_list, adj_counters, n) in order.
        """
        assert len(sentence_list) > 0, "Empty sentence."
        adj_probs = self.calc_adj_probs(self.adj_counters[n])
        assert len(list(adj_probs.keys())[0]) == n, (
            "Non-matching dimension of adj_probs keys and n.")
        cum_neg_log_prob = 0
        if sentence_list[0] != ".":
            sentence_tuple = (".",) + tuple(sentence_list)
        else:
            sentence_tuple = tuple(sentence_list)

        for i in range(0, len(sentence_tuple) - 1):
            prev_phrase = sentence_tuple[max(0, i - n + 2): i + 1]
            assert len(prev_phrase) < n, ("Invalid length of prev_phrase:"
                                          + str(len(prev_phrase)) + ", n: " + str(n))
            curr_word = sentence_tuple[i + 1]

            curr_word_prob = p_func(self,
                                    curr_word,
                                    prev_phrase,
                                    min(len(prev_phrase) + 1, n)
                                    )  # defaults to using p_naive

            ### START YOUR CODE HERE ###
            cum_neg_log_prob -= np.log(curr_word_prob)
            ### END YOUR CODE HERE ###
        return cum_neg_log_prob

    def calc_prob_of_sentence(self, sentence_list, n, p_func=p_laplace):
        """
        Convert a sentence's neg_log_prob to prob.
        """
        ### START YOUR CODE HERE ###
        prob = self.calc_neg_log_prob_of_sentence(sentence_list, n, p_func)
        return np.exp(-prob)
        ### END YOUR CODE HERE ###

    def top_k_adj_starting_with(self, adj_counter, phrase, n, k):
        dict_counter = collections.Counter(self.filter_adj_counter(adj_counter, phrase, n))
        return dict(dict_counter.most_common(k))

    def likeliest_adj_starting_with(self, adj_counter, phrase, n):
        """
        Returns: Tuple with highest num_occurrences starting with `phrase`.
        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> filtered_adj_counter = {("Jimmy", "Li", "is"): 1, ("Jimmy", "Li", "likes"): 100}
        >>> phrase = ("Jimmy", "Li")
        >>> c.likeliest_adj_starting_with(filtered_adj_counter, phrase, 3)
        ("Jimmy", "Li", "likes") # most likely length-3 phrase starting with ("Jimmy", "Li")
        """
        assert isinstance(phrase, tuple), "Phrase is not a tuple."
        subset_word_adj_counter = self.filter_adj_counter(adj_counter, phrase, n)
        return max(subset_word_adj_counter, key=lambda adj: subset_word_adj_counter[adj])

    def generate_sentence(self, k=5, T=1):
        """
        Generate a sentence by sampling amongst the top k candidates when generating each token
        """
        assert k >= 1, "k is not at least 1"

        sentence = ""
        prev_phrase = (".",)
        curr_word = "."
        num_words_in_prev_phrase = 1
        while sentence == "" or curr_word != ".":
            n_iter = min(num_words_in_prev_phrase + 1, self.N)

            top_k_adj_counts = self.top_k_adj_starting_with(self.adj_counters[n_iter], prev_phrase, n_iter, k)
            # print("top_k_adj_counts")

            sampled_adj = self.sample_dict(top_k_adj_counts)

            prev_phrase, curr_word = sampled_adj[:-1], sampled_adj[-1]
            # sentence += (" " if curr_word != "." and sentence else "") + curr_word
            sentence += " " + curr_word
            if num_words_in_prev_phrase < self.N - 1:
                prev_phrase = prev_phrase + (curr_word,)
            else:
                prev_phrase = prev_phrase[1:] + (curr_word,)

            num_words_in_prev_phrase = min(num_words_in_prev_phrase + 1, self.N)

        return sentence

    def generate_likeliest_sentence(self):
        """
        Naive way of generating likeliest sentence.
        First chooses the likeliest word following a period (via `p_naive` technique)
        Then chooses the likeliest word following that word.
        Rinse and Repeat.
        Stop when likeliest word is ".".

        >>> c = N_gram([**comma-separated corpora filepaths**], 2)
        >>> c.generate_likeliest_sentence()
        "we have been a year of the people ."
        """
        return self.generate_sentence(1)