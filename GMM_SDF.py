################################################
# GMM - Asset Pricing
# Author: Jiaying Wu
# Date: Oct. 2024
################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import matplotlib as mpl
from scipy.optimize import minimize
from scipy import stats
import statsmodels.formula.api as sm

### GMM ###
# SDF=1-b*f
# Price of risk: b p*1
# f: Factors or Shocks  p*t
# Re: Excess return of test assets n*t

class GMM_Two_Stage():
    def __init__(self,data_path):
        self.data_path=data_path

        pass
    
    ################################
    # Load Test Assets and Factors #
    ################################
    def data(self):
        # Upload raw data from Ken's website
        TestAssets = pd.read_csv('25_Portfolios_5x5.csv')
        # Time Range: July 1963 - June 2023
        Factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv').iloc[:-11]

        # Merge time series
        df=pd.merge(TestAssets,Factors,on=['jdate']).set_index('jdate')

        self.Factors=df[['Mkt-RF','SMB','HML','RMW','CMA','RF']]

        TestAssets=df.iloc[:,:TestAssets.shape[1]-1]

        self.TestAssets_ex=TestAssets.apply(lambda x: x.sub(self.Factors['RF']))

        # Unit: Percentage -->
        # self.TestAssets_ex=self.TestAssets_ex/100
        # self.Factors=self.Factors/100

        # Number of months, test assets and factors:
        self.t,self.n = self.TestAssets_ex.shape
        _,self.p = self.Factors.shape

        print('Sample Time Period:',self.t)
        print('Test Assets:',self.n)
        print('All Factors:',self.p)



    #####################################
    ### Generalized Method of Moments ###
    #####################################
    def GMM(self, TestAssets_ex, Factors, newey_west_adj=True,nlags=3):
        n,t = TestAssets_ex.shape
        p,_ = Factors.shape

        # First Stage of GMM: given a Equal-weighting matrix
        W_1 = np.eye(n)

        # 
        initial_guess = np.ones(p)
        # Minimization
        b_1 = minimize(ObjFun, initial_guess, args=(TestAssets_ex,Factors,W_1), method='Nelder-Mead').x

        var_b_1 = var_b(b_1,TestAssets_ex,Factors,W_1,Stage=1,newey_west_adj=newey_west_adj,nlags=nlags)

        tstat_b_1 = b_1 / np.sqrt(np.diag(var_b_1))

        # Second Stage of GMM: given a Equal-weighting matrix
        W_2 = np.linalg.inv(S(b_1,TestAssets_ex,Factors))

        # 
        initial_guess2 = b_1

        # Minimization
        b_2 = minimize(ObjFun, initial_guess2, args=(TestAssets_ex,Factors,W_2), method='Nelder-Mead').x

        var_b_2 = var_b(b_2,TestAssets_ex,Factors,W_2,Stage=2,newey_west_adj=newey_west_adj,nlags=nlags)

        std_b_2=np.sqrt(np.diag(var_b_2))

        tstat_b_2 = b_2 / np.sqrt(np.diag(var_b_2))

        p_b_2=2 * (1 - stats.t.cdf(np.abs(tstat_b_2), df=t-p))

        # Pricing Error
        g_2=g_T(b_2,TestAssets_ex,Factors)

        var_g_2=var_g(b_2,TestAssets_ex,Factors,W_2,Stage=2,newey_west_adj=newey_west_adj,nlags=nlags)

        std_g_2=np.sqrt(np.diag(var_g_2))

        tstat_g_2 = g_2 / np.sqrt(np.diag(var_g_2))

        p_g_2=2 * (1 - stats.t.cdf(np.abs(tstat_g_2), df=t-p))

        # Over-identification Test
        J_value=J_stat(b_2,TestAssets_ex,Factors)

        # H0: Moment Conditions = 0
        J_p_value=chi2.pdf(J_value, df=n-p)

        return b_1, tstat_b_1, b_2, tstat_b_2, std_b_2, p_b_2, g_2, std_g_2, p_g_2, J_value, J_p_value



    #######################################################################
    ### Linear Factor Models in Discount Factor Form with Excess Return ###
    ### SDF, GMM
    #######################################################################
    def Price_of_risk(self, regs=[['Mkt-RF'],['Mkt-RF','SMB','HML'],['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],\
                      newey_west_adj=True,nlags=3):
        
        J_stats=[]
        J_pvalues=[]
        SSQE=[]
        MAPE=[]

        # Generate Table Results
        for n_reg,reg in enumerate(regs):
            # n*t
            assets=self.TestAssets_ex.T.to_numpy()
            # p*t
            factors=self.Factors[reg].T.to_numpy()

            b1,b1_t,b2,b2_t,b2_std,b2_p,g2,g2_std,g2_p,J,J_p=GMM_Two_Stage.GMM(self,assets,factors,newey_west_adj=newey_west_adj,nlags=nlags)

            # Making table of price of risk
            b2_p_star=pvalue_func(b2_p)

            # Combine coefficients and p value stars
            b2_cof_p=[str(np.round(b2[i],3))+j for i,j in enumerate(b2_p_star)]

            reg_index=sum([[r,r+"_[std]"] for r in reg],[])

            # Combine coefficients and standard error
            b2_cof_std=sum([[b2_cof_p[i],b2_std[i]] for i,j in enumerate(b2_std)],[])

            tab=pd.DataFrame(b2_cof_std,index=reg_index,columns=["("+str(n_reg+1)+")"])

            J_stats.append(J)
            J_pvalues.append(J_p)


            # Making table of pricing error
            g2_p_star=pvalue_func(g2_p)

            # Combine coefficients and p value stars
            g2_cof_p=[str(np.round(g2[i],3))+j for i,j in enumerate(g2_p_star)]

            alpha_index=["Alpha_model"+str(n_reg+1),"[std]"]

            # Combine coefficients and standard error
            g2_cof_std=[g2_cof_p,g2_std]

            tab2=pd.DataFrame(g2_cof_std,index=alpha_index,columns=self.TestAssets_ex.columns)

            tab2.loc["---"]=["" for i in range(len(self.TestAssets_ex.columns))]

            # sum of squared errors, SSQE: sum square of g_T
            SSQE.append(np.sum(np.square(g2))*100)

            # mean absolute pricing errors, MAPE: average of absolute g_T
            MAPE.append(np.mean(np.abs(g2))*100)

            if n_reg==0:
                Table1=tab.copy()
                Table2=tab2.copy()
            else:
                Table1=pd.concat([Table1,tab],axis=1)
                Table2=pd.concat([Table2,tab2])

            Table1=Table1.replace(np.nan,"")


        Table1.index=["[std]" if "[std]" in j else j for i,j in enumerate(Table1.index)]
        # print(Table1)
        Table1.loc[" "]=["" for i in range(len(regs))]
        Table1.loc["SSQE(%)"]=SSQE
        Table1.loc["MAPE(%)"]=MAPE
        Table1.loc["J statistic"]=J_stats
        Table1.loc["J-Test pvalue"]=J_pvalues

        return Table1, Table2
    


    ###################################
    ### Cross-Sectional Regressions ###
    ###################################
    def Risk_exposure(self,regs=[['Mkt-RF'],['Mkt-RF','SMB','HML'],['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],newey_west_adj=True,nlags=3):

        # Generate Table Results
        for n_reg,reg in enumerate(regs):
            df=pd.merge(self.TestAssets_ex,self.Factors,on="jdate")

            # Running regression on each assets
            assets=self.TestAssets_ex

            # In order to run regression, we should replace the y variable name to be recognizable.
            assets.columns=[a_name.replace(" ", "_") for a_name in assets.columns]

            df.columns=[a_name.replace(" ", "_") for a_name in df.columns]

            df=df.rename(columns={"Mkt-RF":"Mkt"})

            ###################################
            # First Step: Time Series Regression --> Risk Exposure (Betas)
            ###################################
            for n_asset, asset in enumerate(assets.columns):
                
                x=" + ".join(reg)

                # Regression Factors on Excess Return of Assets --> Risk Exposure (Beta)
                if newey_west_adj==True:
                    # print(asset + " ~ " + x)
                    res=sm.ols(formula=asset + " ~ " + x, data=df).fit(
                        cov_type="HAC", cov_kwds={"maxlags": nlags}, use_t=True
                    )
                else:
                    res=sm.ols(formula=asset + " ~ " + x, data=df).fit(
                        use_t=True
                    )

                # print(res.summary2())

                # Exclude the intercept
                beta=res.params[1:]
                beta_std=res.bse[1:]
                beta_p=res.pvalues[1:]

                beta_p_star=pvalue_func(beta_p)

                # Combine coefficients and p value stars
                beta_cof_p=[str(np.round(beta[i],3))+j for i,j in enumerate(beta_p_star)]

                beta_index=sum([["Beta_"+r,"[std]"] for r in reg],[])
                # +f"/Cov(R^e,{r})"

                # Combine coefficients and standard error
                beta_cof_std=sum([[beta_cof_p[i],beta_std[i]] for i,j in enumerate(beta_std)],[])

                tab3=pd.DataFrame(beta_cof_std,index=beta_index,columns=[asset])

                if n_asset==0:
                    table3=tab3.copy()
                else:
                    table3=pd.concat([table3,tab3],axis=1)

            # n*t
            assets_c=self.TestAssets_ex.to_numpy()
            # p*t
            factors_c=self.Factors[reg].to_numpy()

            t,n=assets_c.shape
            t,p=factors_c.shape

            # Cov matrix of F and R
            V=np.cov(np.hstack((factors_c,assets_c)).T)

            # F'F: p*p
            V11=V[0:p,0:p]
            # R'R: n*n
            V22=V[p:,p:]
            # F'R: p*n
            V12=V[0:p,p:]
            # R'F: n*p
            V21=V12.T

            # Risk Exposure: R'F(F'F)^-1 --> n*p
            Beta=V21 @ np.linalg.inv(V11)
            # Add constant term
            X=np.hstack((np.ones((n,1)),Beta))

            Re_mean=assets.to_numpy().mean(axis=0)

            # Residual ε from time series regression
            epsilon=assets_c - factors_c @ Beta.T

            # Sigma Σ = cov(ε_t' ε_t)
            ### Differ from Pollution Premium(2023)
            Sigma=1/(t-p-1)*(epsilon.T @ epsilon)
            # Sigma=np.diag(np.diag(Sigma))

            ######################
            # Second Step: Cross-section Regression --> λ (risk premium) and Alpha (Pricing Error)
            #######################

            # Cochrane(2005) Chp.12.2

            # OLS cross-sectional regression

            # λ
            Lambda_OLS=np.linalg.inv(X.T @ X) @ X.T @ Re_mean
            # α
            Alpha_OLS=Re_mean - X @ Lambda_OLS

            #
            Var_alpha_OLS=(1/t)*(np.eye(n) - Beta @ np.linalg.inv(Beta.T @ Beta) @Beta.T) @ Sigma @ (np.eye(n) - Beta @ np.linalg.inv(Beta.T @ Beta) @Beta.T).T

            # GLS cross-sectional regression

            # λ
            Lambda_GLS=np.linalg.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Re_mean

            # α
            Alpha_GLS=(Re_mean - X @ Lambda_GLS)

            #
            Var_alpha_GLS=(1/t)*(Sigma - Beta @ np.linalg.inv(Beta.T @ np.linalg.inv(Sigma) @ Beta) @ Beta.T)

            # Standard Error
            Alpha_GLS_std=np.sqrt(np.diag(Var_alpha_GLS))
            # t value
            Alpha_GLS_t=Alpha_GLS/np.sqrt(np.diag(Var_alpha_GLS))
            # p value
            Alpha_GLS_p=2 * (1 - stats.t.cdf(np.abs(Alpha_GLS_t), df=n-p-1))
            Alpha_GLS_p_star=pvalue_func(Alpha_GLS_p)

            # Combine coefficients and p value stars
            Alpha_GLS_cof_p=[str(np.round(Alpha_GLS[i],3))+j for i,j in enumerate(Alpha_GLS_p_star)]

            table3.loc['Alpha']=Alpha_GLS_cof_p
            table3.loc['Alpha_[std]']=Alpha_GLS_std
            table3=table3.rename(index={'Alpha_[std]':'[std]'})

            # eq_form=" + ".join([f"Beta^i_{j} * λ_{j}" for j in reg])
            # table3.loc['E[R^e]=α^i + '+ eq_form]=["" for i in range(n)]
            # table3=pd.concat([table3.iloc[-1:],table3.iloc[:-1]])

            table3.loc["Factors:"+",".join(reg)]=["" for i in range(n)]
            table3=pd.concat([table3.iloc[-1:],table3.iloc[:-1]])
            table3.loc['---']=["" for i in range(n)]

            if n_reg==0:
                Table3=table3.copy()
            else:
                Table3=pd.concat([Table3,table3])

        Table3.columns=[c_name.replace("_", " ") for c_name in Table3.columns]
        Table3.index=[i_name.replace("_", " ") for i_name in Table3.index]

        return Table3

            



            


##################################################################################################################################
# Mt
def SDF(b,f):
    return 1 - b.T @ f

# Error: u_t n*t
def u_t(b,Re,f):
    # Hadamard product: Every element R^e_it * M_t
    return SDF(b,f) * Re

# Pricing Error: g_t  n*1
def g_T(b,Re,f):
    return np.mean(u_t(b,Re,f),axis=1)

# Objective Function
def ObjFun(b,Re,f,W):
    # Def. pricing error function
    pricing_error = g_T(b,Re,f)
    # Minimizing the square of
    return pricing_error.T @ W @ pricing_error
##################################################################################################################################


##################################################################################################################################
# əg_t(b)/əb': n*p
# At this linear model, d doesn't depend on b.
def d(b,Re,f):
    n,t = Re.shape
    p,_ = f.shape
    # Special case, make it easy
    return 1/t*(Re @ f.T)

# a=əg_t(b)'/əb @ W
def a(b,Re,f,W):
    return d(b,Re,f).T @ W

# Variance-Covariance Matrix of Errors
def S(b,Re,f):
    return np.cov(u_t(b,Re,f))

# Copy from statsmodels
def weights_bartlett(nlags):
    '''Bartlett weights for HAC

    this will be moved to another module

    Parameters
    ----------
    nlags : int
       highest lag in the kernel window, this does not include the zero lag

    Returns
    -------
    kernel : ndarray, (nlags+1,)
        weights for Bartlett kernel

    '''
    #with lag zero
    return 1 - np.arange(nlags+1)/(nlags+1.)


# Newey-West adjustment for covariance matrix (S)
def NeweyWest_cov(b,Re,f,nlags=None):
    n, t = Re.shape
    residuals = u_t(b, Re, f).T  # t * n residual matrix

    if nlags is None:
        nlags = int(np.floor(4 * (t / 100.)**(2./9.)))

    weights = weights_bartlett(nlags)

    S_neweywest = weights[0] * np.dot(residuals.T, residuals)  # weights[0] just for completeness, is 1

    for lag in range(1, nlags+1):
        s = np.dot(residuals[lag:].T, residuals[:-lag])
        S_neweywest += weights[lag] * (s + s.T)

    return S_neweywest/t


# Variance of b
# Stage: 1st or 2nd stage of GMM estimate, if at 2nd stage, it would use the efficient form.
# newey_west_adj
# nlags
def var_b(b,Re,f,W,Stage,newey_west_adj=False,nlags=None):
    n,t = Re.shape
    p,_ = f.shape
    # Calculate d and a by the value of estimate b
    d_e=d(b,Re,f)
    a_e=a(b,Re,f,W)

    # Whether do neweywest adjustment on covariance matrix
    if newey_west_adj==False:
        S_matrix=S(b,Re,f)
    else:
        S_matrix=NeweyWest_cov(b,Re,f,nlags=nlags)

    # Stage
    if Stage==1:
        return 1/t * np.linalg.inv(a_e @ d_e) @ a_e @ S_matrix @ d_e @ np.linalg.inv(a_e @ d_e)
    elif Stage==2:
        # Optimal
        return 1/t * np.linalg.inv(d_e.T @ np.linalg.inv(S_matrix) @ d_e)

# Variance of g
def var_g(b,Re,f,W,Stage,newey_west_adj=False,nlags=None):
    n,t = Re.shape
    p,_ = f.shape
    # Calculate d and a by the value of estimate b
    d_e=d(b,Re,f)
    a_e=a(b,Re,f,W)

    I=np.eye(W.shape[0])

    # Whether do neweywest adjustment on covariance matrix
    if newey_west_adj==False:
        S_matrix=S(b,Re,f)
    else:
        S_matrix=NeweyWest_cov(b,Re,f,nlags=nlags)

    # Stage
    if Stage==1:
        return 1/t * (I - d_e @ np.linalg.inv(a_e @ d_e) @ d_e) @ S_matrix @ (I - d_e @ np.linalg.inv(a_e @ d_e) @ a_e)
    elif Stage==2:
        # Optimal
        return 1/t * (S_matrix-d_e @ np.linalg.inv(d_e.T @ np.linalg.inv(S_matrix) @ d_e) @ d_e.T)

# Over-identification Test
def J_stat(b,Re,f,newey_west_adj=False,nlags=None):
    n,t = Re.shape
    p,_ = f.shape

    # Whether do neweywest adjustment on covariance matrix
    if newey_west_adj==False:
        S_matrix=S(b,Re,f)
    else:
        S_matrix=NeweyWest_cov(b,Re,f,nlags=nlags)

    return t * (g_T(b,Re,f) @ np.linalg.inv(S_matrix) @ g_T(b,Re,f))
##################################################################################################################################


def pvalue_func(y):
    x = y.copy() 

    x[x < 0.01] = 1
    x[(x >= 0.01) & (x < 0.05)] = 2
    x[(x >= 0.05) & (x < 0.1)] = 3
    x[(x >= 0.1) & (x < 1)] = 4

    mapping = {1: '(***)', 2: '(**)', 3: '(*)', 4: '()'}

    star_replace = np.vectorize(lambda p: mapping.get(p, p))
    p_val = star_replace(x)

    return p_val