# Weight Initalisation

## Linear Layer
Let's say we have linear layer initialized as $W_{(k, d)} \sim \mathcal N(0, \sigma^2)$ and our dataset is $X_{(n, k)}$

Then our output is $Y = XW$, what are the $E[Y], Var[Y]$?
$$y= x_1w_1 + x_2w_2+\dots+x_mw_m$$$$y=t_1+t_2+\dots+t_m \quad t_i=x_iw_i$$
### $E[y]$
$$E[t_i]=E[x_iw_i]=E[x_i]E[w_i]=E[x_i]\cdot 0=0$$
$$E[y]=E[\sum t_i]=\sum E[t_i]=0$$
### $Var[y]$
$$Var[t]=E[t^2]-E[t]^2=E[t^2]=E[x_i^2w_i^2]=E[x_i^2]E[w_i^2]=$$
$$=(Var[x_i]+E[x_i]^2)(Var[w_i]+E[w_i]^2)=Var[x_i]\cdot Var[w_i]$$
therefore:
$$Var[y]=m\cdot Var[t]=m \cdot (Var[x_i]\cdot Var[w_i])$$
### Normalisation
But we want to have $Var[y] = Var[x]$, so we need $Var[w_i] = \frac{1}{m}$
If $W_{(k, d)} \sim \mathcal N(0, 1)$, then $\frac{1}{\sqrt m}W_{(k, d)} \sim \mathcal N(0, \frac{1}{m})$
So the weight initialisation should be here sampling weights from $\mathcal N(0, \frac{1}{m})$.

## Nonlinear Layer
For nonlinear layers, which is linear layer with activation function we use Kaiming Initialisation. The only difference is that instead of $y=x_1w_1+x_2w_2+\dots+x_mw_m$ we have $y=\sigma(z)=\sigma(x_1w_1+x_2w_2+\dots+x_mw_m)$. It means the expected value and variance are different.

This condition holds
$$E[z] = 0$$
But this doesn't
$$Var[z] = mVar(x)Var(w)$$
We don't know whether $E[x]^2 = 0$, so maybe $E[x^2] \neq Var[x]$
Nevertheless we have
$$Var[z] = mE[x^2]Var(w)$$
### ReLU
If $x$ comes from ReLU activation function then 
$$x=ReLU(\hat x)$$

$$E[x^2]=E[\max(0, \hat x)^2]=$$
$$=\int_{-\infty}^\infty \max(0, \hat x)^2\cdot p(\max(0, \hat x)^2)\, d\hat x=$$
$$=\int_{-\infty}^0 \max(0, \hat x)^2\cdot p(\max(0, \hat x)^2)\, d\hat x +\int_{0}^\infty \max(0, \hat x)^2\cdot p(\max(0, \hat x)^2)\, d\hat x=$$
$$=\int_{0}^\infty \max(0, \hat x)^2\cdot p(\max(0, \hat x)^2)\, d\hat x=$$
$$=\int_{0}^\infty \hat x^2 p(\hat x^2)\, d\hat x=$$
$$=\frac{1}{2}E[\hat x^2] \quad \blacksquare$$
So we want $Var[w]=\frac{2}{m}$ to make $Var[z] = Var[\hat x]$
Therefore we multiply $W$ by $\sqrt{\frac{2}{m}}$

