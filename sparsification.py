import math
import matplotlib.pyplot as plt
import numpy as np

def logistic( a, x, n_order = 1, X = []):
    ''' Logistic Function that creates a single hump 
        pattern using the equation:  
        
        F(X) = Xt+1 = a * Xt * (1-Xt)

        This function can be used iteratively using a list of Xt
        or a single initial value of Xt and using Xt+1 for the next
        iteration.

        Where:

        F(X)->    is always in the range of 0 and 1 iff
                Xt is within the range of 0 and 1 and
                a is always less than 4.05

        Xt->  is the population in time or iteration t. 
            The value of Xt must always be between 0,1
            in order to get a hump pattern that defines
            the changes of number of channels within the block.
            Failing to follow the constraint  of 0 and 1 will give negative Xt+1 values.
        
        a->   controls the steepness of the hump or the highest
            number of channels in the block. A value of a
            greater than 4 will cause the max value of F(X) greater
            than 1. 

    '''
    if n_order == 0:
        return X
    else:

        x = a * x * (1-x)
        n_order-=1
        X.append(x)
        return logistic(a,x, n_order=n_order, X = X) 

def logistic2(r,x, i = 0):
    ''' This function is slightly complicated than the first logistic function.
    

        F(X) = Xt+1 = Xt exp[r(1-Xt)]

        r-> tunes the steepness of the nonlinearity.
            Unlike the first logistic function, setting this 
            parameter greater than 4 will not cause Xt+1 to be greater than 1.

        x-> also the population but unlike the first logistic funciton,
            setting this parameter greater than 1 will attract F(X)  to 0
            but will not completely drag the value to 0 unless this parameter
            is large enough.  

    '''

    return r * x * (1-x)#x**(r*(1-x))

def second_order_logistic(a, x):
    ''' Function to get the second order or 2 period logistic function. 
        This Function gives a dual hump nonlinear plot.
        
        Xt+2 = F[F(Xt)] = F[Xt+1] = a * Xt+1 * (1-Xt+1)

    '''
    return logistic2(a,logistic2(a,x))

def fibonacci(depth):

    f = []
    for i in range(1,depth+1):
        num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
        den = 2**i * (math.sqrt(5))
        f.append(int(num/den))
    return(f)

def naive_block_channels_variation(blocks, depth = 5, ratio = 0.618):
    block_list = {}
    for idx, block in enumerate(blocks):
        depth_ = depth
        ratio_ = ratio 
        while depth_ > 0:
            val = int( (block * ratio_ * (1 - ratio_))*100)
            block_list['block_%d_layer%d'%(idx,depth-depth_)] = val
            ratio_ = 3.414 * ratio_ * (1 - ratio_)
            depth_ -= 1
    return block_list   
    
def build_blocks(num_blocks = 5, block_depth = 5):
    initial_ratio = 0.618
    block_dict = naive_block_channels_variation(fibonacci(num_blocks), block_depth)
    

    


# X = np.linspace(0, 1, num = 10)
# print(X)
# vals = []
# for x in X:
#     v = logistic2(3.9601 ,x)
#     print(v)
#     vals.append(v)
# # print(len(vals))
# ratio = 0.618
# for b in fibonacci(4):
#     vals = logistic(b,ratio,n_order=10)

vals = naive_block_channels_variation(fibonacci(5))
for key, val in vals.items():
    print(key,val)

# plt.plot(vals)
# plt.show()
