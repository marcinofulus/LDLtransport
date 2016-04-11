import numpy as np 
from scipy.sparse import dia_matrix
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt

def phi_od_t(t, phi_max):
    "time dependent fraction of leaky junctions"
    phi_max= phi_max/0.0107826845403
    x_list = [0, 60*5, 60*30, 60*60, 60*120]
    y_list = [0.000483806767739,0.002266*phi_max,0.009701*phi_max,  0.0107826845403*phi_max,  0.0107826845403*phi_max]
    if t<=5*60.:
        phi_70=5e-4
        a=(y_list[1]-phi_70)/(5*60.)
        phi = a*t+phi_70
        return phi
    
    if t<=30*60.:
        a=(y_list[2]-y_list[1])/(25.*60)
        b=y_list[1]-5*60*a
        phi = a*t+b
        return phi 
    
    if t<=60*60.:
        a=(y_list[3]-y_list[2])/(30.*60)
        b=y_list[2]-30*60*a
        phi = a*t+b
        return phi 
 
    return 0.0107826845403*phi_max
        


def phi(WSS = 1.79, dPmmHg = 70):
    "fraction of leaky junctions"
    Rcell = 15e-3                              # mm
    area=.64                                   # mm^2
    SI = 0.38*np.exp(-0.79*WSS) + 0.225*np.exp(-0.043*WSS)
    MC = 0.003797* np.exp(14.75*SI)
    LC = 0.307 + 0.805 * MC 
    phi = (LC*np.pi*Rcell**2)/(area)
    return( phi) 

def Klj(w=14.3e-6,phi=5e-4):
    "permability  in m^2"
    Rcell = 15e-3 # mm
    return ( (w**2/3.)*(4.*w*phi)/Rcell * (1e-6) )

def Kend(w=14.3e-6,phi=5e-4):
    "permability  w m^2"
    Kend_70mmHg =3.22e-21
    Knj = Kend_70mmHg - Klj() # at 70mmHg
    return Knj + Klj(w,phi)

def sigma_end(phi=5e-4,w=14.3*1e-6,r_m = 11e-6):
    a = r_m/w
    Kend_70mmHg =3.22e-21
    Knj = Kend_70mmHg - Klj() # at 70mmHg
    sigma_lj = 1-(1-3/2.*a**2+0.5*a**3)*(1-1/3.*a**2)
    return  1 - ((1-sigma_lj)*Klj(phi=phi))/(Knj+Klj(phi=phi))

def porosity_end(w=14.3e-6,phi=5e-4):
    Rcell = 15e-3
    return 4*w/Rcell*phi

def Diffusivity(w=14.3e-6,phi=5e-4,r_m = 11e-6):
    "Diffusivity  w um^2/s"
    R_cell = 15e-6 # m
    a=r_m/w
    D_lumen=2.867e-11
    return D_lumen*(1-a)*(1.-1.004*a+0.418*a**3-0.169*a**5)*4*w/R_cell*phi*1e-3*1e12

class Parameters(object):

    
    def calculate_filration(self,dPmmHg):
        mmHg2Pa = 133.3
        dP = mmHg2Pa*dPmmHg
        Rw = [L_*mu_/K_ for L_,K_,mu_ in zip(self.L,self.K,self.mu)]
        self.Vfiltr = dP/sum(Rw)
    
    def update(self, phi):
        self.sigma_end = sigma_end(phi=phi,w=14.3*1e-6,r_m = 11e-6)
        self.Kend = Kend(w=14.3e-6,phi=phi)
        self.K[0] = self.Kend*1e6
        self.calculate_filration(dPmmHg=self.dPmmHg)
        self.V = [ self.Vfiltr*1e6]*4
        self.sigma[0] = self.sigma_end
        self.D[0]=Diffusivity(w=14.3e-6,phi=phi,r_m = 11e-6)
        self.porosity[0]=porosity_end(w=14.3e-6,phi=phi)
        self.D1[0] = self.D[0]/self.porosity[0]
        self.V1 = [V_/porosity_ for V_,porosity_ in zip(self.V,self.porosity)]
    
    def __init__(self,WSS=1.79,dPmmHg=70):
        """ Change units to mikrometers
        Class can be initialized with a value of WSS in Pa
        """
        self.nazwy       = [  'endothel' , 'intima', 'IEL'   ,'media' ]
        self.D           = [    ]
        self.V           = [   ]
        self.sigma       = [  ] 
        self.L           = [   2.        , 5   ,    2.   , 161.   ]
        self.k_react     = [  ]
        self.mu          = [   0.72e-3   , 0.72e-3 , 0.72e-3 , 0.72e-3 ]
        self.porosity    = [   0.002     , 0.8025  ,  0.003  , 0.258   ]
        self.dPmmHg=dPmmHg
        
    def get_params(self):
        return (self.D,self.V,self.sigma,self.L,self.k_react)
    
    def calculate_missing_values(self,WSS=1.79,dPmmHg=70, phi_max=5e-4):
        self.phi = phi_max+phi(WSS=WSS,dPmmHg=dPmmHg)-5e-4
        self.sigma_end = sigma_end(phi=self.phi,w=14.3*1e-6,r_m = 11e-6)
        self.Kend = Kend(w=14.3e-6,phi=self.phi)
        self.K[0] = self.Kend*1e6 
        self.calculate_filration(dPmmHg=dPmmHg)
        
        self.D = [D_*1e6 for D_ in self.D]
        self.V = [ self.Vfiltr*1e6]*4 
        self.sigma[0] = self.sigma_end
        self.D[0]=Diffusivity(w=14.3e-6,phi=self.phi,r_m = 11e-6)
        self.porosity[0]=porosity_end(w=14.3e-6,phi=self.phi)
        
class LDL_Parameters_4L(Parameters):
    """ Our parameters"""

    def __init__(self,WSS=1.79,dPmmHg=70, compressed = True, phi_max=5e-4):
        """ Change units to mikrometers
        Class can be initialized with a value of WSS in Pa
        """
        super(LDL_Parameters_4L, self).__init__(WSS=WSS,dPmmHg=dPmmHg )
        if  not compressed:
            self.D           = [   5.7e-12   , 3.68e-6   , 1.066e-8     , 5e-8     ]
            self.V           = [   2.3e-2 ]*4
            self.sigma       = [   0.9888    , 0.7998    , 0.8051       , 0.8836   ] 
            self.k_react     = [   0.        , 0.        , 0.           , 3.197e-4 ]
            self.K           = [   3.22e-15  , 3.907e-11 , 1.363543e-13 , 2e-12    ]
            self.L           = [   2.        , 5.0       , 2.           , 161.     ]
            self.porosity    = [   0.002     , 0.802     , 0.003        , 0.258    ]
            self.calculate_missing_values(WSS, dPmmHg,phi_max=phi_max)
            
        elif dPmmHg ==160:
            self.D           = [   5.7e-12   , 8.263e-7 , 1.066e-8     , 5e-8     ]
            self.V           = [   2.3e-2 ]*4
            self.sigma       = [   0.9888    , 0.9876   , 0.8051       , 0.8836   ] 
            self.k_react     = [   0.        , 0.       , 0.           , 3.197e-4 ]
            self.L           = [   2.        , 2.       , 2.           , 161.     ]
            self.K           = [   3.22e-15  , 7.72e-12 , 1.363543e-13 , 2e-12    ]
            self.porosity    = [   0.002     , 0.5401   , 0.003        , 0.258    ]
            self.calculate_missing_values(WSS, dPmmHg, phi_max=phi_max)
            
        elif dPmmHg ==120:
            self.D           = [   5.7e-12   , 1.08e-6  , 1.066e-8     , 5e-8     ]
            self.V           = [   2.3e-2 ]*4
            self.sigma       = [   0.9888    , 0.9789   , 0.8051       , 0.8836   ] 
            self.k_react     = [   0.        , 0.       , 0.           , 3.197e-4 ]
            self.L           = [   2.        , 2.3      , 2.           , 161.     ]
            self.K           = [   3.22e-15  , 9.91e-12 , 1.363543e-13 , 2e-12    ]
            self.porosity    = [   0.002     , 0.5862   , 0.003        , 0.258    ]
            self.calculate_missing_values(WSS, dPmmHg,phi_max=phi_max)
            
        elif dPmmHg == 70:
            self.D           = [   5.7e-12   , 3.68e-6   , 1.066e-8     , 5e-8     ]
            self.V           = [   2.3e-2 ]*4
            self.sigma       = [   0.9888    , 0.7998    , 0.8051       , 0.8836   ]
            self.k_react     = [   0.        , 0.        , 0.           , 3.197e-4 ]
            self.K           = [   3.22e-15  , 3.907e-11 , 1.363543e-13 , 2e-12    ]
            self.L           = [   2.        , 5.0       , 2.           , 161.     ]
            self.porosity    = [   0.002     , 0.802     , 0.003        , 0.258    ]
            self.calculate_missing_values(WSS, dPmmHg, phi_max=phi_max)
            
        self.D1 = [D_/porosity_ for D_,porosity_ in zip(self.D,self.porosity)]
        self.V1 = [V_/porosity_ for V_,porosity_ in zip(self.V,self.porosity)]
        self.k_react1 = [k_/porosity_ for k_,porosity_ in zip(self.k_react,self.porosity)]
            

class LDL_Sim(object):
   
    def __init__(self, pars):
        self.pars = pars
        self.c_st = None
    def discretize(self,N=2000):
        self.N = N
        k = np.ones(N)
        v = np.ones(N)
        Dyf = np.ones(N)
        k1 = np.ones(N)
        v1 = np.ones(N)
        Dyf1 = np.ones(N)
        D,V,sigma,L,k_react = self.pars.get_params()
        D1=self.pars.D1
        V1=self.pars.V1
        k_react1=self.pars.k_react1
        
        l = np.sum(L) # dlugosc ukladu
        self.l = l
        self.x=np.linspace(0,l,N)
        
        layers=[0]+list( np.ceil( (N*(np.cumsum(L)/sum(L)))).astype(np.int32) )
        self.layers=layers
        for i,(l1,l2) in enumerate(zip(layers[:],layers[1:])):
            k[l1:l2] = k_react[i]
            v[l1:l2] = (1.0-sigma[i])*V[i]
            Dyf[l1:l2] = D[i]
            k1[l1:l2] = k_react1[i]
            v1[l1:l2] = (1.0-sigma[i])*V1[i]
            Dyf1[l1:l2] = D1[i]
        dx2_1 = (N-1)**2/l**2
        dx_1 = (N-1)/l

        diag_l = np.ones(N)*(np.roll(Dyf1,-1)*dx2_1)
        diag   = np.ones(N)*(-2.*Dyf1*dx2_1 - k1 + v1*dx_1)
        diag_u = np.ones(N)*(np.roll(Dyf1,1)*dx2_1 - np.roll(v1,1)*dx_1)

        # Layer's junctions
        for j in layers[1:-1]:
            diag[j] = v[j-1]-v[j+1]-(Dyf[j-1]+Dyf[j+1])*dx_1
            diag_l[j-1] = Dyf[j-1]*dx_1
            diag_u[j+1] = Dyf[j+1]*dx_1
        #BC
        diag[0] = 1
        diag[-1] = 1
        diag_u[0+1] = 0
        diag_l[0-2] = 0
   
        self.L = dia_matrix((np.array([diag_l,diag,diag_u]),np.array([-1,0,1])), shape=(N,N))
        
    def change_discretization(self, t,N=2000):
        t_max = self.Nsteps *self.dt
        phi_max=self.phi_max/0.0107
        phi_t = phi_od_t(t, self.phi_max)
        self.pars.update(phi_t)
        self.discretize(N)
          
    def solve_stationary(self,bc=[1,0]):
        b = np.zeros(self.N)
        b[0],b[-1] = bc
        L = self.L.tocsr()
        self.c_st = sp.sparse.linalg.linsolve.spsolve(L,b)
        por = np.ones(self.N)
        layers=self.layers
        for i,(l1,l2) in enumerate(zip(layers[:],layers[1:])):
            por[l1:l2] = self.pars.porosity[i]
        self.c_st2=self.c_st*por
        
    def essential_boundary_conditions(self,u):
        u[0] = 1.0*(1-self.dt)
        u[-1] = 0.0

        
    def solve_time(self,dt =9.0, Nsteps=500, sps=100):
    
        # initial condition
        N = self.N
        u = np.zeros(N)
        self.sps = sps     
        self.dt = dt
        self.Dt = dt*sps
        self.Nsteps = Nsteps
        I=scipy.sparse.eye(N,N,format='csc')
        if self.pars.dPmmHg ==70:
            L1 = self.L.tocsc()
        self.Tlst=[]
        self.Tlst2=[]
        self.cInt_lst=[]
        self.essential_boundary_conditions(u)
        print "Nsteps:",Nsteps

        for i in range(Nsteps):
            if self.pars.dPmmHg !=70:
                self.change_discretization(i*dt,N)
                L1 = self.L.tocsc()
            if not i%sps:
                self.Tlst.append(list(u))
                por = np.ones(self.N)
                layers=self.layers
                for i,(l1,l2) in enumerate(zip(layers[:],layers[1:])):
                    por[l1:l2] = self.pars.porosity[i]
                u2=u*por
                self.Tlst2.append(list(u2))
               
            self.essential_boundary_conditions(u)
            u = sp.sparse.linalg.linsolve.spsolve(I-dt*L1,u)
            self.cInt_lst.append(np.sum(u)*(self.l/(self.N-1)))
        print "Results saved in table Tlst."
             
    def plot_c(self,yrange=(0,0.2),xrange=(0,214),filename=None, kolor='red', alpha=0.2, style='-', linewidths=2):
        
        i1,i2 = int(xrange[0]/self.l*self.N),int(xrange[1]/self.l*self.N)
        plt.plot(self.x[i1:i2],self.c_st[i1:i2],color=kolor,linewidth=linewidths, ls=style)
        plt.ylim( *yrange)
        plt.xlim( *xrange)
        
        L=self.pars.L
        d=[0]+np.cumsum(self.pars.L).tolist()
        colors=['m','g','b','w']
        for i,(l1,l2) in enumerate(zip(d[:],d[1:])):
            plt.bar([l1,],yrange[1],l2-l1, color=colors[i], linewidth=0.3,  alpha=alpha)
        
        plt.grid(True,axis='y', which='major')
        plt.xlabel(r"$x \left[\mu m\right]$")
        plt.ylabel(r"$c(x)$")
        if filename!=None:
            plt.savefig(filename)
            
    def plot_c2(self,yrange=(0,0.2),xrange=(0,214),filename=None, kolor='red', alpha=0.2, style='-', linewidths=2):
        i1,i2 = int(xrange[0]/self.l*self.N),int(xrange[1]/self.l*self.N)
        plt.plot(self.x[i1:i2],self.c_st2[i1:i2],color=kolor,linewidth=linewidths, ls=style)
        plt.ylim( *yrange)
        plt.xlim( *xrange)
        
        L=self.pars.L
        d=[0]+np.cumsum(self.pars.L).tolist()
        colors=['m','g','b','w']
        for i,(l1,l2) in enumerate(zip(d[:],d[1:])):
            plt.bar([l1,],yrange[1],l2-l1, color=colors[i], linewidth=0.3,  alpha=alpha)   
            
        plt.grid(True,axis='y', which='major')
        plt.xlabel(r"$x \left[\mu m\right]$")
        plt.ylabel(r"$c(x)$")
        if filename!=None:
            plt.savefig(filename)


def LDL_simulation_4L(wss=1.79,dPmmHg=70, bc=[1,0.0047],compressed=True, phi_max=5e-4):
    pars = LDL_Parameters_4L(WSS=wss,dPmmHg=dPmmHg,compressed=compressed, phi_max=phi_max)
    sim = LDL_Sim(pars)
    sim.phi_max=phi_max
    sim.dPmmHg=dPmmHg
    l = np.sum(pars.L)
    sim.discretize(130*l)
    sim.solve_stationary(bc=bc)
    return sim

#Integrate function 
def c_integrate(dx=20, x0=2,xk=170, ppum=130,  concentration=[], points=[]):
    if len(points)==0:
        points= np.arange(dx,xk,dx)
        c_sum=[np.sum(concentration[(x+x0)*ppum-dx*ppum:(x+x0)*ppum]/(ppum*dx)) for x in points]
    else:
        c_sum=[]
        points.insert(0,x0)
        for i,(l1,l2) in enumerate(zip(points[:],points[1:])):
            c_sum.append(np.sum(concentration[((l1)*ppum):l2*ppum]/(ppum*(l2-l1)))) 
        points.remove(x0)
        
    return points, c_sum