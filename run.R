library(INLA)
library(Matrix)

### load files
Z_mat <- as(Matrix::readMM('Z.mtx'), "Matrix") # RE design matrix Z
Q_precision <- as(Matrix::readMM('Ginv.mtx'), "Matrix") # Precision matrix Q, this is the Cmatrix in the INLA documenation https://inla.r-inla-download.org/r-inla.org/doc/latent/z.pdf
y_outcome <- as.numeric(scan("pheno.txt")) # the phenotype

### fit INLA model
df.data <- data.frame(y = y_outcome, id.z = 1:length(y_outcome))
obj.inla <- INLA::inla(
    y ~ f(id.z, model = "z", Z = Z_mat, Cmatrix = Q_precision),
    data=df.data,
    verbose=TRUE,
    control.predictor = list(compute=TRUE)
    )

saveRDS(obj.inla, 'inla.result.rds')