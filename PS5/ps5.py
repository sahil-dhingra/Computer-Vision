"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.cov = 0.1 * np.eye(4)
        self.dt = np.eye(4)
        self.dt[0,2] = 1
        self.dt[1,3] = 1
        self.sigma_dt = Q
        self.mt = np.zeros((2,4))
        self.mt[0,0] = 1
        self.mt[1,1] = 1       
        self.sigma_mt = R

    def predict(self):
        self.state = np.dot(self.dt, self.state)
        self.cov = np.dot(np.dot(self.dt, self.cov), self.dt.T) + self.sigma_dt
        

    def correct(self, meas_x, meas_y):
        self.y = np.array([meas_x, meas_y])
        K = np.dot(np.dot(self.cov, self.mt.T), 
                   np.linalg.inv(np.dot(self.mt, np.dot(self.cov, self.mt.T)) + self.sigma_mt))
        self.state = self.state + np.dot(K, (self.y - np.dot(self.mt, self.state)))
        self.cov = np.dot(np.eye(4) - np.dot(K, self.mt), self.cov)
        

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame       
        h, w = template.shape[:2]
        if h%2==1:
            self.template=self.template[:-1,:,:]
        if w%2==1:
            self.template=self.template[:,:-1,:]
        h_ = int(h/2)
        w_ = int(w/2)
        h, w = frame.shape[:2]
        
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_ = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        mse = [self.get_error_metric(template_, frame_[i-h_:i+h_, j-w_:j+w_])
                                    for i in range(h_,h-h_) for j in range(w_,w-w_)]
        mse = np.reshape(np.array(mse), (h-2*h_,w-2*w_))
        p_zx = np.exp(-mse/(2*self.sigma_exp))
         
        particles = np.argsort(np.ravel(p_zx))[::-1][:self.num_particles]
        self.particles = np.array([h_+particles%(h-2*h_),h_+particles//(h-2*h_)]).T
        particles = self.particles
        particles.T[0] = np.clip(particles.T[0], w_, w-w_-1)
        particles.T[1] = np.clip(particles.T[1], h_, h-h_-1)
        self.particles = np.array(particles, dtype = 'int')

        self.particles0 = self.particles.copy()
        
        # Initialize your particles array. Read the docstring.
        self.weights = (1/self.num_particles)*np.ones(self.num_particles)
        # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.


    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        mse = np.mean(np.square(template - frame_cutout))
        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        n = self.get_weights().size
        sample_ind = np.random.choice(n, n, p=self.get_weights())
        particles = np.take(self.particles, sample_ind, axis=0)
        
        return particles
        return NotImplementedError

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_ = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        frame_ = np.average(frame, axis=2)
        template_ = np.average(self.template, axis=2)
        h, w = template_.shape[:2]
        
        h_ = int(h/2)
        w_ = int(w/2)
        h, w = frame.shape[:2]
        
        mse = np.array([self.get_error_metric(template_, frame_[i-h_:i+h_, j-w_:j+w_])
                                                            for j, i in self.particles])

        p_zx = np.exp(-mse/(2*self.sigma_exp))
        self.weights = p_zx/sum(p_zx)
        self.particles = self.resample_particles()
        self.particles0 = self.particles.copy()
        
        u_gauss = np.random.normal(loc=0, scale=self.sigma_dyn, size=self.num_particles)
        v_gauss = np.random.normal(loc=0, scale=self.sigma_dyn, size=self.num_particles)
        particles = np.array([self.particles.T[0] + u_gauss, self.particles.T[1] + v_gauss]).T
        particles.T[0] = np.clip(particles.T[0], w_, w-w_-1)
        particles.T[1] = np.clip(particles.T[1], h_, h-h_-1)
        self.particles = np.array(particles, dtype = 'int')
        

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        # Dots
        for i in range(self.num_particles):
            center_w = (self.particles[i][0], self.particles[i][1])
            cv2.circle(frame_in, center_w, radius=1, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
        
        # Rectangle
        h, w = self.template.shape[:2]
        y_top = int(x_weighted_mean - h/2)
        y_bottom = int(x_weighted_mean + h/2)
        x_left = int(y_weighted_mean - w/2)
        x_right = int(y_weighted_mean + w/2)
        frame_in = cv2.rectangle(frame_in, (y_top, x_left), (y_bottom, x_right), color=(0, 255, 0), thickness=1)
        
        # Circle
        center_circle = (int(x_weighted_mean), int(y_weighted_mean))
        dist = np.linalg.norm(self.particles - center_circle, axis=1)
        radius_part = int(np.average(dist, weights = self.weights))
        cv2.circle(frame_in, center_circle, radius=radius_part, color=(255, 0, 0), thickness=2, lineType=8, shift=0)



class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """

        super(AppearanceModelPF, self).process(frame)
        
        h, w = self.template.shape[:2]
        
        h_ = int(h/2)
        w_ = int(w/2)
        h, w = frame.shape[:2]

        mse = np.array([np.mean(np.square(self.template-frame[i-h_:i+h_, j-w_:j+w_,:]))
                                                            for j, i in self.particles0])    
            
        particles_ind = np.argsort(mse)[2:6]
        particles = self.particles0[particles_ind]
        mse = mse[particles_ind]
        
        u_mean = int(np.average(particles.T[0]))
        v_mean = int(np.average(particles.T[1]))
        if np.std(self.particles.T[0])<75 and np.std(self.particles.T[0])<75:
            self.template = ((1-self.alpha)*self.template + (self.alpha)*frame[v_mean-h_:v_mean+h_, u_mean-w_:u_mean+w_,:]).astype(np.uint8)
           
       


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        self.scale = 1
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        super(MDParticleFilter, self).process(frame)
        particle = self.particles0[np.argmax(self.weights)]
        j, i = particle
        scale = np.random.normal(loc=self.scale, scale=0.1, size=10)
        templates = [cv2.resize(self.template, None, fx=k, fy=k) for k in scale]
        for i in range(10):
            if templates[i].shape[0]%2==1:
                templates[i] = templates[i][:-1,:,:]
            if templates[i].shape[1]%2==1:
                templates[i] = templates[i][:,:-1,:]
        template_size = [(int(template.shape[0]/2), int(template.shape[1]/2)) for template in templates]
        print(template_size)
        mse = np.array([np.mean(np.square(templates[k]-frame[i-template_size[k][0]:i+template_size[k][0], j-template_size[k][1]:j+template_size[k][1],:])) for k in range(10)])    
        self.scale = scale[np.argmin(mse)]
        self.template = cv2.resize(self.template, None, fx=self.scale, fy=self.scale)