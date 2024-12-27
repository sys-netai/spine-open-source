import numpy as np

class G_Noise(object):
    def __init__(self, mu , sigma, explore=40000,theta=0.1,mu2=0.0,mode="exp",eps=1.0,step=0.3):
        self.epsilon = eps
        self.mu = mu
        self.explore = explore
        self.sigma = sigma
        self.mu2 = mu2
        self.theta = theta
        self.noise = 0
        self.cnt = 0
        self.step = step
        self.mode = mode

    def show(self):
        return self.noise

    def __call__(self,point):
        if self.explore!=None:
            if self.mode=="exp":
                if self.epsilon <= 0.05:
                    noise = self.epsilon * (self.sigma * np.random.randn(1))
                    self.noise = noise
                    #self.noise=np.zeros_like(self.mu)
                else:
                    self.epsilon -= 1/self.explore
                    noise = self.epsilon * (self.sigma * np.random.randn(1))
                    self.noise = noise
            else:
                self.cnt += 1
                if self.cnt >=self.explore:
                    self.sigma -= self.step*self.sigma
                    self.cnt = 0
                if self.sigma <= 0.1:
                    self.sigma = 0.1
                noise = self.sigma*np.random.randn(1)
                self.noise = noise
        else:
            noise = (self.sigma * np.random.randn(1))
            self.noise = noise

        return self.noise

    def reset(self):
        pass


class Random_Noise(object):
    def __init__(self, explore=40000,eps=1.0):
        self.epsilon = eps
        self.explore = explore
        self.noise = 0
        print("get")

    def show(self):
        return self.noise

    def __call__(self, point):
        if self.explore != None:
            if self.epsilon <= 0.05:
                pass
            else:
                self.epsilon -= 1/self.explore
            # print(self.epsilon)
            if np.random.uniform(0,1) < self.epsilon:
                noise = np.random.uniform(-1, 1) - point
            else:
                noise = 0
            self.noise = np.array([noise])
        else:
            noise = np.random.uniform(-1, 1)
            self.noise = np.array([noise])

        return self.noise

    def reset(self):
        pass


class OU_Noise:
    def __init__(self, mu, sigma, theta=.15, dt=0.01, x0=None,exp=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.eps = 1.0
        self.exp = exp
        self.reset()

    def show(self):
        return self.x_prev

    def __call__(self,point):
        if self.exp!=None:
            self.dt -= 1/self.exp
            if self.dt<=0.01:
                self.dt=0.01
            self.sigma -= 1/self.exp
            if self.sigma<=0.3:
                self.sigma=0.3

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

