# Classification and Inpainting using BCM

* [Theory] (#theory)


## Theory

The model at the basis of this work is called BCM (Bienenstock, Cooper and Munro) theory that refers to the synaptic modification first proposed in 1982.
This theory of synaptic plasticity is based on three postulates:

* The change in synaptic weights <img src="https://latex.codecogs.com/png.image?\dpi{110}\frac{dw_{i}}{dt}" title="https://latex.codecogs.com/png.image?\dpi{110}\frac{dw_{i}}{dt}" /> is proportional to presynaptic activity <img src="https://latex.codecogs.com/png.image?\dpi{110}x_{i}" title="https://latex.codecogs.com/png.image?\dpi{110}x_{i}" />;
* This change is also proportional to a non-monotonic function <img src="https://latex.codecogs.com/png.image?\dpi{110}\phi" title="https://latex.codecogs.com/png.image?\dpi{110}\phi" /> of the postsynaptic activity y. It has two different behaviours according to the postsynaptic activity: it decreses for low y and increases for higher y;
* The modification threshold, indicated with <img src="https://latex.codecogs.com/png.image?\dpi{110}\theta" title="https://latex.codecogs.com/png.image?\dpi{110}\theta" /> and corresponding to <img src="https://latex.codecogs.com/png.image?\small&space;\dpi{110}\frac{\mathrm{d}&space;w_i}{\mathrm{d}&space;t}>0" title="https://latex.codecogs.com/png.image?\small \dpi{110}\frac{\mathrm{d} w_i}{\mathrm{d} t}>0" />, is itself a superlinear function of the history of postsynaptic activity y.

### Mathematical Formulation od BCM

The original BCM equation is defined by:

<img src="https://latex.codecogs.com/png.image?\dpi{110}y&space;=&space;\sum_{i}w_ix_i&space;\\\\\indent&space;\frac{\mathrm{d}&space;w_i}{\mathrm{d}&space;t}&space;=&space;y(y-\theta)x_i-\epsilon&space;w_i,&space;\\\\\indent&space;\theta&space;=&space;E[y/y_0]&space;" title="https://latex.codecogs.com/png.image?\dpi{110}y = \sum_{i}w_ix_i \\\\\indent \frac{\mathrm{d} w_i}{\mathrm{d} t} = y(y-\theta)x_i-\epsilon w_i, \\\\\indent \theta = E[y/y_0], " />

while a more recent formula has been drawn from Law and Cooper in 1994:

<img src="https://latex.codecogs.com/png.image?\dpi{110}y&space;=&space;\sigma\Biggl(\sum_{i}w_ix_i\Biggl),&space;\\\\\indent&space;\frac{\mathrm{d}w_i}{\mathrm{d}&space;t}&space;=&space;\frac{y(y&space;-&space;\theta)x_i}{\theta},&space;\\\\\indent&space;\theta&space;=&space;E[y^{2}]&space;" title="https://latex.codecogs.com/png.image?\dpi{110}y = \sigma\Biggl(\sum_{i}w_ix_i\Biggl), \\\\\indent \frac{\mathrm{d}w_i}{\mathrm{d} t} = \frac{y(y - \theta)x_i}{\theta}, \\\\\indent \theta = E[y^{2}] " />.


