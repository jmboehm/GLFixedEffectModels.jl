library("alpaca")
library("foreign")

df <- read.table("/Users/johannes.boehm/Dropbox/Github/GLFixedEffectModels.jl/dataset/Cigar.csv", header = TRUE, sep = ",", dec = ".")
df[["Sales2"]] <- (df[["Sales"]]-mean(df[["Sales"]]))/sd(df[["Sales"]])
df[["Sales2"]] <- df[["Sales2"]] - min(df[["Sales2"]] )
formula <- Sales2 ~ Price | State
ctrl <- feglm.control(trace = 2L)
mod <- feglm(formula, df, family = poisson(), control = ctrl)
summary(mod)

formula2 <- Sales ~ Price
summary(glm(formula2, df, family = poisson()))

summary(lm(formula2, df))




formula       = Sales ~ Price | State
data          = df
family        = poisson()
beta.start    = NULL
D.alpha.start = NULL
control       = ctrl