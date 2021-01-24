import numpy as np

class Kalman():
    def __init__(self,dt,U,std):

        #dt: time intravel in which readings taken
        #U: acceleration in x and y direction [ux,uy] as list
        #std : variance in x direction,y direction, and variance in acceleration as list [std_x,std_y,std_acc]


        # initialize [[px,py,vx,vy]] as zero
        self.X = np.zeros(shape=(4,1))
        self.U = np.array(U).reshape(2,-1)
        self.dt = dt
        # error in measurements x and y (ie, std deviation of measurements)
        self.xm_std,self.ym_std,self.std_acc = std[0],std[1],std[2]

        # Define the State Transition Matrix A
        self.A = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # input control matrix
        self.B = np.array([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # since we are tracking only position of a moving object we have (we are not tracking velocity)
        self.H = np.array([[1,0,0,0],[0,1,0,0]])

        #process covariance matric  # for now we initialize as an identity matrix
        self.P = np.eye(self.A.shape[0])

        #process noise covariance matrix  // Dynamic noise
        #standard deviation of position as the standard deviation of acceleration multiplied by dt**2/2
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc**2

        #measurement noise covariance matrix  // Measurement noise
        self.R = np.array([[self.xm_std**2,0],
                           [0, self.ym_std**2]])

        self.process_noise = 0
        self.measurement_noise = 0

        
    def predict(self):
        
        # predict the distance and velocity
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.U) + self.process_noise

        # predicted process cov matrix
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        
        return self.X[0:2]

    def update(self,Xm):
        Xm = np.array(Xm).reshape(2,1)
        
        # calculate kalaman gain
        # K = P * H'* inv(H*P*H'+R)
        denominator = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(denominator)) #shape: (4,2)
        

        # measurments
        C = np.eye(Xm.shape[0])
        Xm = np.dot(C,Xm) + self.measurement_noise

        # update the predicted_state to get final prediction of iteration and process_cov_matrix
        self.X = self.X + np.dot(K,(Xm - np.dot(self.H,self.X)))

        #update process cov matrix
        self.P = (np.eye(K.shape[0]) - np.dot(np.dot(K,self.H),self.P))



        return self.X[0:2]