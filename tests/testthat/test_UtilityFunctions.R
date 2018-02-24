context("Testing utility functions")

test_that(
  desc = "checkProb works as expected",
  code = {
    # general case
    ninstances <- 2
    a <- matrix(
      c(
        0.3, 0.4, 0.3,
        0.2, 0.6, 0.2
      ),
      nrow = ninstances
    )
    classes <- c("a", "b", "c")
    colnames(a) <- classes
    
    r <- checkProb(a, ninstances, classes)
    expect_identical(r, a)
    
    b <- a[, c(2, 1, 3)]
    r <- checkProb(b, ninstances, classes)
    expect_identical(r, a)
    
    # one instance case
    ninstances <- 1
    a <- matrix(c(0.3, 0.4, 0.3), nrow = ninstances)
    colnames(a) <- c("a", "b", "c")
    r <- checkProb(a, ninstances, classes)
    expect_identical(r, a)
    
    b <- matrix(c(0.3, 0.3, 0.4), nrow = ninstances)
    colnames(b) <- c("a", "c", "b")
    r <- checkProb(b, ninstances, classes)
    expect_identical(r, a)
    
    # prob is not a matrix
    expect_error(checkProb(1, 1, classes))
    # prob is a matrix but has not column names
    ninstances <- 2
    a <- matrix(
      c(
        0.3, 0.4, 0.3,
        0.2, 0.6, 0.2
      ),
      nrow = ninstances
    )
    expect_error(checkProb(a, ninstances, classes))
    # 
    colnames(a) <- c("a", "b", "c")
    expect_error(checkProb(a, ninstances = 1, classes))
    # classes mismach
    colnames(a) <- c("a", "b", "c")
    expect_error(checkProb(a, ninstances, c("a", "b", "c", "d")))
    colnames(a) <- c("a", "b", "d")
    expect_error(checkProb(a, ninstances, c("a", "b", "c")))
    a <- matrix(
      c(
        0.1, 0.4, 0.3, 0.1,
        0.2, 0.5, 0.2, 0.1
      ),
      nrow = ninstances
    )
    colnames(a) <- c("a", "b", "c", "d")
    expect_error(checkProb(a, ninstances, c("a", "b", "c")))
  }
)