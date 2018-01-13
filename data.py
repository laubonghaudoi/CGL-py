import numpy as np
from scipy.sparse import spdiags


class Data(object):
    '''
    Read course files and return a data set object. Course pairs are splited into
    3 folds, for training, validating and testing.

    According to Section 2 in the paper, the number of courses is denoted with $n$
    and dimension of universal concept space is denoted with $p$. We replace them
    with `num_courses` and `num_dimensions` to improve readability.
    '''

    def __init__(self, opt):
        '''
        Each data object has 4 member variables:
            1. `opt`: Configuration object
            2. `X`: Bag-of-concepts representations of all courses, an array of size
                    [num_courses, num_dimensions]
            3. `chunks`: A length 3 dict, as 3-fold data set consisting of raw course
                    pairs and labels
            4. `T`: Course-level triplets created with raw course pairs, a list with
                    3 numpy arrays as the train, validation and test set
        '''
        self.opt = opt

        concepts, links = self._read_dataset()

        self.X = self._normalize(concepts)

        pairs, y = self._get_pairs(links, len(concepts))
        self.chunks = self._split_pairs(pairs, y)
        self.T = self._get_chunks(self.chunks)

    def _read_dataset(self):
        '''
        Read .lsvm file into `X` and .link file into `links`

        Each row of *.lsvm is the bag-of-words feature for a course
        <course-id> <word>:<count> <word>:<count> ... <word>:<count>

        Each row of *.link is a pair of courses with a prerequisite relationship
        <prerequisite-course-id> <postrequisite-course-id>

        Args:
            `opt`: Configuration object

        Return:
            `concepts`: Length `num_courses` dict, every key is a feature id
            `links`: Length [num_links, 2] list, all prerequisite and postrequisite
                    links between courses
        '''
        # Read *.lsvm file into features of each course
        lines = open(self.opt.course_file).readlines()
        concepts = []
        for line in lines:
            line = line.split()
            bag_of_words = {}
            # The first number in every line of data is course id, skip it
            for word in line[1:]:
                idx = word.split(':')
                bag_of_words[int(idx[0])] = int(idx[1])
            concepts.append(bag_of_words)

        # Read *.link files into course links
        lines = open(self.opt.prereq_file).readlines()
        links = []
        for line in lines:
            course = line.split()
            links.append([int(course[1]), int(course[0])])

        return concepts, links

    def _normalize(self, concepts):
        '''
        Convert `concepts` into numpy array and normalize by courses.

        Args:
            `concepts`: Bags of concepts of all courses, a length `num_courses` dict.

        Return:
            `X`: Normalized numpy array form of all concepts
        '''
        # Count total number of features
        num_courses = len(concepts)
        num_dimensions = 0
        for course in concepts:
            n = max(course.keys())
            if n > num_dimensions:
                num_dimensions = n

        # Convert list into numpy array
        X = np.zeros((num_courses, num_dimensions))

        for course_id, course in enumerate(concepts):
            for feature in course:
                X[course_id, feature - 1] = course[feature]

        # Normalize X
        # The matlab code uses `row_norm = max=(d,1)` and I do not understand
        norm = np.sqrt(np.sum(X**2, axis=1, keepdims=True)).transpose()
        diags = np.array([0])
        temp = spdiags(norm, diags, num_courses, num_courses).toarray()
        # We use np.linalg.lstsq as a substitute for matlab \ operator
        X = np.linalg.lstsq(temp, X)[0]

        return X

    def _get_pairs(self, links, num_courses):
        '''
        Generate all possible pairs of courses and add labels

        Args:
            `links`: Length `num_links` list
            `n`: Number of concepts

        Return:
            `pairs`: Possible pairs between two courses, [num_pairs, 2] list
            `y`: Label of pairs, [num_pairs, 1] list
        '''

        num_pairs = num_courses * (num_courses - 1)
        pairs = [[]] * num_pairs
        k = 0

        for i in range(num_courses):
            for j in range(num_courses):
                if i != j:
                    # It is tricky here, since Matlab's index starts from 1 whereas
                    # Python from 0, we need to plus 1 to all indices
                    pairs[k] = [i + 1, j + 1]
                    k += 1

        y = [0] * num_pairs
        for i, r in enumerate(pairs):
            if r in links:
                y[i] = 1
            else:
                y[i] = -1

        return pairs, y

    def _split_pairs(self, pairs, y):
        '''
        Split the data into 3 folds for cross validation.

        Args:
            `pairs`: Outputs from `self.get_pairs`
            `y`: Outputs from `self.get_pairs`

        Return:
            `chunk`: 3-fold data set, dict with following keys:
                'pairs': Course pairs
                'labels': ±1 labels
                'num_pairs': Number of pairs
        '''
        pairs = np.array(pairs)
        y = np.array(y)

        num_pairs = len(pairs)
        chunks = []
        for k in range(3):
            chunk = {}
            index = np.arange(k, num_pairs, 3)

            chunk['pairs'] = pairs[index]
            chunk['labels'] = y[index]
            chunk['num_pairs'] = index.shape[0]

            chunks.append(chunk)

        return chunks

    def _generate_triplets(self, chunk):
        '''
        Generate all triplets based on pairs and their labels, where (i, j) are
        positive samples and (i, k) are negative.

        In the `pairs` array, each line is a course pair in the form of:
        <prerequisite-course-id> <postrequisite-course-id>, we pick up the positive
        and negative samples


        Args:
            `chunk`: A dict with course pairs and labels

        Return：
            `triplets`: Triplets (i, j, k) based on the pairs, a [num_pairs, 3]
                        size array.
        '''
        Triplets = []
        # Convert from list to numpy arrays
        pairs = np.array(chunk['pairs'], dtype=int)
        y = np.array(chunk['labels'], dtype=int)

        # Pairs that with positive or negative labels
        pos = pairs[y == 1, :]
        neg = pairs[y == -1, :]

        # For every course
        for i in list(set(pairs[:, 0])):
            # The postrequisite course of this course
            pos_pool = pos[pos[:, 0] == i, 1]
            # If no correspondence, pass
            if len(pos_pool) == 0:
                continue
            # Negative samples of postrequisite courses
            neg_pool = neg[neg[:, 0] == i, 1]

            # Compose an array where each line is an (i, j, k) sample,
            # representing (course, positive, negative)
            a, b = np.meshgrid(pos_pool, neg_pool)
            combo = np.array([a.flatten(), b.flatten()]).transpose()

            triplet = np.append(
                i * np.ones((combo.shape[0], 1), dtype=int), combo, axis=1)
            Triplets.append(triplet)

        Triplets = np.vstack(Triplets)

        return Triplets

    def _get_chunks(self, chunks):
        '''
        Make all chunks of course pairs into course triplets

        Args:
            `chunks`: A dict with course pairs and labels

        Return:
            `T`: A list of course triplet arrays
        '''
        T = []
        for i in range(3):
            triplets = self._generate_triplets(chunks[i])
            T.append(triplets)

        return T
