
data(polarization, package='ggcleveland')
pol <- polarization
set.seed(123)
pol$conc3 <- jitter( pol$concentration^(1/3), factor = 10 )
plot(babinet ~ conc3, data = pol, pch=19, col='gray70', cex=1.1)

library(mgcv)
a <-gam(babinet ~ s(conc3), data=pol) 

# Kalogridis
# https://github.com/ioanniskalogridis/Smoothing-splines
source('https://raw.githubusercontent.com/ioanniskalogridis/Smoothing-splines/main/Huber')
y <- pol$babinet[order(pol$conc3)]
x <- sort(pol$conc3)
fit.huber1 <- huber.smsp(x = x, y = y, k  = 1.345, interval = c(1e-16, 1)) 

# Oh et al (robust smoothing)
library(fields)
oh1 <- qsreg(x=x, y=y, maxit.cv=100) 
pr <- predict(oh1, x)
si <- mad( y - pr )
oh2.0 <- qsreg(x=x, y=y, sc = si, maxit.cv = 100) 
lam<- oh2.0$cv.grid[,1]
tr<- oh2.0$cv.grid[,2]
lambda.good<- max(lam[tr>=15])
oh2 <- qsreg(x=x, y=y, sc = si, lam = lambda.good) 


# RBF
library(RBF)
y <- pol$babinet[order(pol$conc3)]
x <- sort(pol$conc3)
n <- length(y)
nh <- 10
hh <- seq(.1, .3, length=nh) # previous search over (.05, 1), narrowed here
cvbest <- +Inf
rmspe <- rep(NA, nh)
for(i in 1:nh) {
  # leave-one-out CV loop
  preds <- rep(NA, n)
  for(j in 1:n) {
    tmp <- try( backf.rob(y ~ x, point = x[j],
                          windows = hh[i], epsilon = 1e-6,
                          degree = 1, type = 'Tukey', subset = c(-j) ))
    if (class(tmp)[1] != "try-error") {
      preds[j] <- rowSums(tmp$prediction) + tmp$alpha
    }
  }
  pred.res <- preds - y
  tmp.re <- RobStatTM::locScaleM(pred.res, na.rm=TRUE)
  rmspe[i] <- tmp.re$mu^2 + tmp.re$disper^2
  if( rmspe[i] < cvbest ) {
    jbest <- i
    cvbest <- rmspe[i]
    print('Record')
  }
  print(c(i, rmspe[i]))
}
bandw <- hh[jbest]
tmp <- backf.rob(y ~ x, windows=bandw, degree=1, type='Tukey', point=x)


# S-penalized splines
# https://github.com/msalibian/PenalizedS
source("https://raw.githubusercontent.com/msalibian/PenalizedS/master/pen-s-functions.R")
lambdas <- seq(1e-4, 10, length = 100) 
p <- 3
y <- pol$babinet[order(pol$conc3)]
x <- sort(pol$conc3)
n <- length(x)
num.knots <- max(5, min(floor(length(unique(x))/4), 35))
knots <- quantile(unique(x), seq(0, 1, length = num.knots + 2))[-c(1, (num.knots + 2))]

xpoly <- rep(1, n)
for (j in 1:p) xpoly <- cbind(xpoly, x^j)
xspline <- outer(x, knots, "-")
xspline <- pmax(xspline, 0)^p
X <- cbind(xpoly, xspline)
# penalty matrix
D <- diag(c(rep(0, ncol(xpoly)), rep(1, ncol(xspline))))

NN <- 500  # max. no. of iterations for S-estimator
# NNN no. of initial candidates for the S-estimator
cc <- 1.54764
b <- 0.5
tmp.s <- pen.s.rgcv(y = y, X = X, D = D, lambdas = lambdas, num.knots = num.knots, 
                    p = p, NN = NN, cc = cc, b = b, NNN = 50)

# Fig 1
pdf(file='example-smooth.pdf', bg='transparent')
plot(babinet ~ conc3, data = pol, pch=19, col='gray70', cex=1.1, 
     xlab=expression(Conc^{1/3}), ylab='Babinet')
lines(pol$conc3, fitted(a), type='l', lwd=3, col='gray30')
lines(x, fit.huber1$fitted, lwd=3, lty=2, col='blue3')
lines(x, predict(oh2, x), lwd=5, lty=3, col='red3')
lines(x, predict(oh2.0, x), lwd=3, lty=4, col='green4')

dev.off()

# Fig 2
pdf(file='example-pen-kern.pdf', bg='transparent')
plot(babinet ~ conc3, data = pol, pch=19, col='gray70', cex=1.1, 
     xlab=expression(Conc^{1/3}), ylab='Babinet')
lines(x, tmp.s$yhat, lwd=3, lty=1, col='red3')
lines(x, as.numeric(tmp$prediction + tmp$alpha), lwd=3, lty=2, col='blue3')
dev.off()
