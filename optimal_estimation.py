# -*- coding: utf-8 -*-
__version__ = '1.1'  # 10-01-2015
__author__ = 'Rene Preusker, rene.preusker@fu-berlin.de'

#Todo catch math errors
import numpy as np
import collections
from numpy.linalg import inv as npinv

EPSIX = np.finfo(float).resolution

#if you like *besser* naming, change it here
result= collections.namedtuple('result','x j conv ni g a sr cost')

def inverse(inn):
    try:
        out=npinv(inn)
    except np.linalg.LinAlgError:
        # Is this always smart?
        #or better to flag ..
        out=np.zeros_like(inn)
    return out

def right_inverse(inn):
    return np.dot(inn.T, inverse(np.dot(inn, inn.T)))

def left_inverse(inn):
    return np.dot(inverse(np.dot(inn.T, inn)), inn.T)

#bit useless
#Todo refactor
def clipper(a, b, x):
    return np.clip(x, a, b)

###Gauss Newton Operator
def newton_operator(a, b, x, fnc, dfnc):
    '''

    :param a: lower limit of x np.array with 1 dimension
    :param b: upper limit of x np.array with same length  as a
    :param x: state vector
    :param fnc: function, accepting x as input
    :param dfnc: jacobian function  of fnc
    :return: cnx (clipped) root of fnc for the linear case, last y=fnc(x), last increment of x
    '''
    y = fnc(x)
    k = dfnc(x)
    ki = left_inverse(k)
    incr_x = np.dot(ki, y)
    cnx = clipper(a, b, x - incr_x)
    return cnx, y, k, incr_x, None, None
def newton_cost(x,fnc):
    '''
    L2 norm
    :param x:
    :param fnc:
    :return:
    '''
    y=fnc(x)
    cost = np.dot(y.T, y)
    return cost
def newton_ret_err_cov_i(x,dfnc):
    return None
def newton_ret_err_cov(x,dfnc):
    return None
def newton_gain_aver_cost_sr(y, x, k):
    '''
    Calculates Gain, averagiong kernel matrix and cost
    :param y:
    :param x:
    :param k:
    :return:
    '''
    # gain matrix
    gain = left_inverse(k)
    # averaging kernel
    aver = np.identity(x.size)
    # cost function
    cost = np.dot(y.T, y)
    return gain, aver, cost, None

###Gauss Newton with measurement error
def newton_operator_with_se(a, b, x, fnc, dfnc, sei):
    '''

    :param a: lower limit of x np.array with 1 dimension
    :param b: upper limit of x np.array with same length  as a
    :param x: state vector
    :param fnc: function, accepting x as input
    :param dfnc: jacobian function  of fnc
    :param sei: inverse of measurement error co-variance
    :return: cnx (clipped) root of fnc for the linear case, last y=fnc(x), last increment of x, last
            retrieval error co.-variance
    '''
    y = fnc(x)
    k = dfnc(x)
    #print 'sei',sei
    kt_sei = np.dot(k.T, sei)
    #print 'kt_sei',kt_sei
    ret_err_cov_i = (np.dot(kt_sei, k))
    #print 'ret_err_cov_i',ret_err_cov_i
    ret_err_cov = inverse(ret_err_cov_i)
    #print 'ret_err_cov',ret_err_cov
    kt_sei_y = np.dot(kt_sei, y)
    #print 'kt_sei_y',kt_sei_y
    incr_x = np.dot(ret_err_cov, kt_sei_y)
    #print 'nx',x - incr_x
    cnx = clipper(a, b, x - incr_x)
    #print 'cnx',cnx
    return cnx, y, k, incr_x, ret_err_cov_i, ret_err_cov
def newton_se_ret_err_cov_i(x,k,sei):
    kt_sei = np.dot(k.T, sei)
    # inverse retrieval error co-variance
    kt_sei_k = np.dot(kt_sei, k)
    return kt_sei_k
def newton_se_ret_err_cov(x,k,sei):
    return inverse(newton_se_ret_err_cov_i(x,k,sei))
def newton_se_cost(x,fnc,sei):
    y=fnc(x)
    cost = np.dot(y.T, np.dot(sei, y))
    return cost
def newton_se_gain_aver_cost_sr(y, x, k, sei,ret_err_cov=None):
    #retrieval error co-varince
    if ret_err_cov is None:
        ret_err_cov= newton_se_ret_err_cov(x,k,sei)
    # gain matrix
    gain = np.dot(ret_err_cov, np.dot(k.T, sei))
    # averaging kernel
    # aver=np.dot(gain,k)
    aver = np.identity(x.size)
    # cost function
    cost = np.dot(y.T, np.dot(sei, y))
    return gain, aver, cost, ret_err_cov

###Optimal estmation with Gauss Newton
def optimal_estimation_gauss_newton_operator(a, b, x, fnc, dfnc, sei, sai, xa):
    '''

    :param a: lower limit of x np.array with 1 dimension
    :param b: upper limit of x np.array with same length  as a
    :param x: state vector
    :param fnc: function, accepting x as input
    :param dfnc: jacobian function  of fnc
    :param sei: inverse of measurement error co-variance
    :param sai: inverse of prior error co-variance
    :param xa: prior
    :return: cnx (clipped) optimal solution for  fnc-1 for the linear case, last y=fnc(x), last increment of x, last
            retrieval error co.-variance
    '''
    y = fnc(x)
    k = dfnc(x)
    kt_sei = np.dot(k.T, sei)
    kt_sei_k = (np.dot(kt_sei, k))
    # inverse retrieval error co-variance
    ret_err_cov_i = sai + kt_sei_k
    # retrieval error co-variance
    ret_err_cov = inverse(ret_err_cov_i)
    kt_sei_y = np.dot(kt_sei, y)
    sai_dx = np.dot(sai, xa - x)
    incr_x = np.dot(ret_err_cov, kt_sei_y - sai_dx)
    cnx = clipper(a, b, x - incr_x)
    return cnx, y, k, incr_x, ret_err_cov_i, ret_err_cov
def oe_ret_err_cov_i(x,k,sei,sai):
    kt_sei = np.dot(k.T, sei)
    kt_sei_k = np.dot(kt_sei, k)
    # inverse retrieval error co-variance
    ret_err_cov_i = sai + kt_sei_k
    return ret_err_cov_i
def oe_ret_err_cov(x,k,sei,sai):
    return inverse(oe_ret_err_cov_i(x,k,sei,sai))
def oe_cost(x,xa,fnc,sei,sai):
    y=fnc(x)
    cost = np.dot((xa - x).T, np.dot(sai, xa - x)) + \
        np.dot(y.T, np.dot(sei, y))
    return cost
def oe_gain_aver_cost_sr(y, x, k, xa, sei, sai,ret_err_cov=None):
    #retrieval error co-varince
    if ret_err_cov is None:
        ret_err_cov= oe_ret_err_cov(x,k,sei,sai)
    # gain matrix
    gain = np.dot(ret_err_cov, np.dot(k.T, sei))
    # averaging kernel
    aver = np.dot(gain, k)
    # cost function
    cost = np.dot((xa - x).T, np.dot(sai, xa - x)) + \
        np.dot(y.T, np.dot(sei, y))
    return gain, aver, cost, ret_err_cov


def numerical_jacoby(a, b, x, fnc, nx, ny, delta):
    '''

    :param a:
    :param b:
    :param x:
    :param fnc:
    :param nx:
    :param ny:
    :param delta:
    :return: Jacobian of fnc
    '''
    # very coarse but sufficient for this excercise
    #dx = np.array((b - a) * delta)
    dx=(b - a) * delta
    jac = np.zeros((ny, nx))  # zeilen zuerst, spalten später!!!!!!!
    for ix in range(nx):
        dxm = x * 1.
        dxp = x * 1.
        dxm[ix] = dxm[ix] - dx[ix]
        dxp[ix] = dxp[ix] + dx[ix]
        dyy = fnc(dxp) - fnc(dxm)
        for iy in range(ny):
            jac[iy, ix] = dyy[iy] / dx[ix] / 2.
    return jac

def my_optimizer(
        a,
        b,
        y,
        func,
        fparams,
        jaco,
        jparams,
        xa,
        fg,
        sei,
        sai,
        eps,
        maxiter,
        method=2,
        delta=0.001,
        epsx=EPSIX * 2.,
        epsy=0.000001,
        full=False):
    '''

    a:      lower limit of state
    b:      upper limit of state
    y:      measurement
    func:   function to be inverted
    jaco:   that returns the jacobian of func
    sei:    inverse of measurement error covariance matrix
    sai:    inverse of prior error covariance matrix
    xa:     prior state
    fg:     first guess state
    delta:  dx=delta*(b-a) to be used for jacobian
    epsy:   if norm(func(x)-y)  < epsy, optimization is stopped
    epsx:   if max(new_x^2/(b-a)) < epsx, optimization is stopped
    eps:    if (x_i-x_i+1)^T # S_x # (x_i-x_i+1)   < eps * N_x, optimization
            is stopped. This is the *original* as, e.g. proposed by Rodgers
    params: are directly passed to func (e.g. geometry, special temperatures, profiles aerosols ...)
   jparams: are directly passed to jaco (e.g. geometry, special temperatures, profiles aerosols ...)
    method: optimizer (0:  pure GaussNewton, 1: gaussnewton with measurement error  2: optimal
            estimisation with Gauss Newton optimizer )

    '''

    ### put some curry to the meet
    ###
    # function to root-find
    def fnc(x):
        return func(clipper(a, b, x), fparams) - y

    # numerical derivation of fnc (central differential ...)
    if jaco is None:
        def dfnc(x):
            return numerical_jacoby(a, b, x, fnc, x.size, y.size, delta)
    else:
        def dfnc(x):
            return jaco(x, jparams)

    if method == 0:
        # Newton Step
        def operator(x):
            return newton_operator(a, b, x, fnc, dfnc)
        def diagnose(x, y, k):
            return newton_gain_aver_cost_sr(y, x, k)
        def cost(x):
            return newton_cost(x,fnc)
    elif method == 1:
        # Newton with measurement error
        def operator(x):
            return newton_operator_with_se(a, b, x, fnc, dfnc, sei)
        def diagnose(x, y, k, sr=None):
            return newton_se_gain_aver_cost_sr(y, x, k, sei, sr)
        def cost(x):
            return newton_se_cost(x,fnc,sei)
    elif method == 2:
        # Optimal Estimation
        def operator(x):
            return optimal_estimation_gauss_newton_operator(a, b, x, fnc, dfnc, sei, sai, xa)
        def diagnose(x, y, k, sr=None):
            return oe_gain_aver_cost_sr(y, x, k, xa, sei, sai, sr)
        def cost(x):
            return oe_cost(x,xa,fnc,sei,sai)


    def normy(inn):
        return (inn * inn).mean()

    #def normx(inn):
    #    return ((inn * inn) / (b - a)).max()

    def norm_error_weighted_x(ix, sri):
        # see Rodgers for details
        # ix : increment of x = x_i -x_i+1
        # sri: inverse of retrieval error co-variance
        return np.dot(ix.T, np.dot(sri, ix))

    # prior as first guess ...
    if fg is None:
        xn = xa
    else:
        xn = fg

    ### Do the iteration
    yn=fnc(xn)
    ii, conv = 0, False
    while True:
        ii += 1
        # Iteration step
        xn, yn, kk, ix, sri, sr = operator(xn)

        # Rodgers convergence criteria
        if method > 0:
            if norm_error_weighted_x(ix, sri) < eps * ix.size:
                conv=True
                break

        # only aplicable if *no* prior knowledge is used
        if method < 2:
            if normy(yn) < epsy:
                conv=True
                break

        # if x doesnt change ,  stop
        #if normx(ix) < epsx:
        #    print 'epsx',epsx,ix,xn
        #    conv=True
        #    break

        # if maxiter is reached,  no-converge and stop
        if ii > maxiter:
            conv = False
            break

    if full is False:
        return xn, kk, conv, ii, None,None,None,None
    elif full == 'fast':
        #take the last-but-one
        gg, av, co, sr = diagnose(xn, yn, kk, sr)
        return xn, kk, conv, ii, av, gg, sr, co
    elif full is True:
        #calculate latest yn, kk, sr
        yn=fnc(xn)
        kk=dfnc(xn)
        #calculate diagnose quantities
        gg, av, co, sr = diagnose(xn, yn, kk)
        return xn, kk, conv, ii, av, gg, sr, co
def my_inverter(func, a, b, **args):
    '''
    This invertes (solves) the following equation:
    y=func(x,params)
    and returns a function which is effectively
    the inverse of func
    x=func⁻¹(y,fparams=params)

    mandatory INPUT:
           a = lower limit
           b = upper limit
       func  = function to be inverted
               y=func(x)
     optional INPUT:
       eps   = convergence criteria when iteration is stopped (Rodgers),xtol=0.001
       xad   = default prior x
       fgd   = default first guess x
       sad   = default prior error co-variance
       sed   = default measurement error co-variance
   methodd   = default operator (2=optimal estimation)
  maxiterd   = default maximum number of iterations (2=optimal estimation)
      jaco   = function that returns jacobian of func
               jacobian = jaco(x)

    OUTPUT:
       func_inverse    inverse of func
    '''
    for kk in ['sed', 'sad', 'xad', 'fgd', 'eps', 'methodd', 'jaco']:
        if kk not in args:
            args[kk] = None
    if args['methodd'] is None:
        args['methodd'] = 2
    if args['eps'] is None:
        args['eps'] = 0.01
    if args['fgd'] is None:
        args['fgd'] = (a + b) / 2.
    def func_inverse(
            yy,
            se=args['sed'],
            sa=args['sad'],
            xa=args['xad'],
            eps=args['eps'],
            fg=args['fgd'],
            method=args['methodd'],
            jaco=args['jaco'],
            maxiter=20,
            full=False,
            fparams=None,
            jparams=None):
        '''
        Input:
            yy   = measurement
            se   = measurement error co-variance matrix
            sa   = prior error co-variance matrix
            xa   = prior
            fg   = first guess
          method = 0-newton 1-newto+se  2-optimal_estimation
        linesrch = False  no additionaö line search True
                   aditional linesearch (only if nonlinear, more than quadratic)
          jaco   = function that returns jacobian of func
         fparams = additional parameter for func
         jparams = additional parameter for jaco

        Output:
            x    = retrieved state
            sr   = retrieval error co-variance matrix
            a    = averaging kernel matrix
            g    = gain matrix
            j    = jacobian
            ni   = number of iterations
            conv = convergence (True or False)
            cost = cost function
        '''
        if method == 0:
            sei = np.zeros((yy.size, yy.size))
            sai = np.zeros((a.size, a.size))
        elif method == 1:
            sai = np.zeros((a.size, a.size))
            sei = inverse(se)
        elif method == 2:
            sai = inverse(sa)
            sei = inverse(se)
        else:
            sei = np.zeros((yy.size, yy.size))
            sai = np.zeros((a.size, a.size))

        if isinstance(yy, list):
            yyy = np.array(yy)
        elif isinstance(yy, float):
            yyy = np.array([yy])
        else:
            yyy = yy
        xxx, jjj, ccc, nnn, aaa, ggg, sss, cst = my_optimizer(  a=a, b=b
                                                              , y=yyy, xa=xa, fg=fg
                                                              , sei=sei, sai=sai
                                                              , eps=eps, maxiter=maxiter, method=method
                                                              , func=func, fparams=fparams, jaco=jaco, jparams=jparams
                                                              , full=full)
        if full is False:
            return xxx
        else:
            return result(xxx, jjj,  ccc, nnn, ggg, aaa, sss, cst)
    return func_inverse


def test():
    '''
    Tested are 3 cases 'funca', 'funcb', 'funcc'
    a:  linear R^2 --> R^3
    b:  nonlinear R^2 --> R^3
    c:  nonlinear R^3 --> R^2  (only OE)
    '''

    # lower bound of state
    A = np.array([0.1, 0.1, 0.1])
    # upper bound of state
    B = np.array([10, 10, 10])
    AA = {'a': A[0:2], 'b': A[0:2], 'c': A}
    BB = {'a': B[0:2], 'b': B[0:2], 'c': B}

    # SE measurement error covariance
    SEa = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 100.]]) * .1
    SEb = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]) * 10.
    SEc = np.array([[10., 0.], [0., 10.]]) * 0.1
    SE = {'a': SEa, 'b': SEb, 'c': SEc}

    # SA apriori error covariance
    SAa = np.array([[1., 0.], [0., 1.]]) * 1.
    SAb = np.array([[1., 0.], [0., 1.]]) * 1.
    SAc = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) * 100.
    SA = {'a': SAa, 'b': SAb, 'c': SAc}

    # XA prior knowledge
    XA = {'a': np.array([3.7, 5.6]),
          'b': np.array([3.7, 5.6]),
          'c': np.array([3.7, 5.6, 8.5])
          }

    # XT to test
    XT = {'a': np.array([3.5, 6.5]),
          'b': np.array([3.5, 6.5]),
          'c': np.array([3.5, 6.5, 5.8])
          }

    def funca(x, *args, **kwargs):
        '''
        simple linear R^2-->R^3 test function
        '''
        return np.array([13. + 6*x[0] + 4*x[1]
                          , 2. - 3*x[0] + 2*x[1]
                          ,        x[0] - 5*x[1]
                          ])

    def funcb(x,*args,**kwargs):
        '''
        simple non-linear R^2-->R^3 test function
        '''
        return np.array([ 13 + 6*x[0] + 4*x[1] + 0.7*np.power(x[0]*x[1],4)
                          , 2 - 3*x[0] + 2*x[1] +   np.sqrt(x[0])*np.log(x[1])
                          ,       x[0] - 5*x[1] -   np.sqrt(x[0]*x[1])
                          ])
    def funcc(x,*args,**kwargs):
        '''
        simple linear R^3-->R^2 test function.
        '''
        return np.array([ 13+6*x[0]+4*x[1]-2*x[2]
                          ,2-3.*x[0]+5.*x[1]+7*x[2]
                          ])

    FUNC = {'a': funca, 'b': funcb, 'c': funcc}

    method = ('Newton', 'Newton+SE', 'OE')

    for func_key in ['a', 'b', 'c']:
        print '-' * 30
        print func_key * 30
        print '-' * 30
        # func_key,rr='b',range(0,3)
        # func_key,rr='a',range(0,3)
        print 'XA', XA[func_key], '--> YA:', FUNC[func_key](XA[func_key])
        print 'SE'
        print SE[func_key]
        print 'SA'
        print SA[func_key]
        yt = FUNC[func_key](XT[func_key])
        inv_func = my_inverter(FUNC[func_key], AA[func_key], BB[func_key])

        print
        print 'Test with x =', XT[func_key], '-->  y=', yt
        for i, meth in enumerate(method):
            if i != 2 and func_key == 'c':
                continue
            erg = inv_func(yt,full=True, sa=SA[func_key], se=SE[func_key], xa=XA[func_key], eps=0.001, method=i, maxiter=100)
            print '    retrieved X: ', erg.x
            if i > 0:
                print '    diag avKern: ', [erg.a[j, j] for j in range(erg.a.shape[0])]
                print '    diag ret.er: ', [erg.sr[j, j] for j in range(erg.sr.shape[0])]
            print '         nitter: ', erg.ni
            print '           cost: ', erg.cost
            print '    f(retrie X): ', FUNC[func_key](erg.x)

        yt[-1] = yt[-1] + 1
        print
        print 'Test with x =', XT[func_key], '-->  disturbed y=', yt
        for i, meth in enumerate(method):
            if i != 2 and func_key == 'c':
                continue
            erg = inv_func(yt,full=True,sa=SA[func_key],se=SE[func_key],xa=XA[func_key],eps=0.001,method=i)
            #for _ in range(1000): dum=inv_func(yt,full=True,sa=SA[func_key],se=SE[func_key],xa=XA[func_key], eps=0.001,method=i)
            print '    retrieved X: ', erg.x
            if i > 0:
                print '    diag avKern: ', [erg.a[j, j] for j in range(erg.a.shape[0])]
                print '    diag ret.er: ', [(erg.sr[j, j]) for j in range(erg.sr.shape[0])]
            print '         nitter: ', erg.ni
            print '           cost: ', erg.cost
            print '    f(retrie X): ', FUNC[func_key](erg.x)


if __name__ == '__main__':
    test()
