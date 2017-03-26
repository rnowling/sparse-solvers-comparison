#include <stdio.h>
#include <time.h>

#include <petscksp.h>

static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -matrix <matrix_file> : matrix to use\n\
  -rhs    <vec_file> : RHS vector for linear system\n\n";

Mat load_my_matrix(char* filename)
{
    PetscInt n_rows, n_cols, n_nnzs;
    PetscErrorCode ierr;
    Mat A;
    char buffer[255];
    FILE* matrix_file = fopen(filename, "r");
    
    // initialize data structures
    ierr = MatCreate(PETSC_COMM_WORLD,&A);
    CHKERRQ(ierr);

    // skip comments
    do {
        fgets(buffer, 255, matrix_file);
    } while(buffer[0] == '%');

    // read matrix dimensions
    if (sscanf(buffer, "%d %d %d", &n_rows, &n_cols, &n_nnzs) != 3)
    {
        printf("Failed to parse matrix entry\n");
        printf("Line: '%s'\n", buffer);
        fclose(matrix_file);
        exit(1);
    }

    printf("Reading a %d x %d matrix\n", n_rows, n_cols);
    
    // Creates a sequential or parallel AIJ matrix based on number of processors in the communicator
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);

    // set up matrix dimensions
    ierr = MatSetSizes(A, n_rows, n_cols, n_rows, n_cols);
    CHKERRQ(ierr);
    
    // allocate memory
    ierr = MatSetUp(A);
    CHKERRQ(ierr);

    // read in data
    printf("Reading data\n");
    while(fgets(buffer, 255, matrix_file) != NULL && !feof(matrix_file)) {
        PetscInt row, col;
        PetscScalar value;

        if (sscanf(buffer, "%d %d %lf", &row, &col, &value) != 3)
        {
            printf("Failed to parse matrix entry\n");
            printf("Line: '%s'\n", buffer);
            fclose(matrix_file);
            exit(1);
        }

        // convert from 1-indexing to 0-indexing
        ierr = MatSetValue(A, row-1, col-1, value, INSERT_VALUES);
        CHKERRQ(ierr);
    }

    fclose(matrix_file);

    printf("Read data\n");

    // finalize data structures
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    return A;
}

Vec load_my_vector(char* filename)
{
    PetscInt col = 0;
    PetscInt n_cols = 0;
    PetscErrorCode ierr;
    Vec b;
    char buffer[255];
    FILE* vector_file = fopen(filename, "r");
    
    // initialize data structures
    ierr = VecCreate(PETSC_COMM_WORLD,&b);
    CHKERRQ(ierr);

    // read vec dimension
    fgets(buffer, 255, vector_file);
    if (sscanf(buffer, "%d", &n_cols) != 1)
    {
        printf("Failed to parse vector size\n");
        printf("Line: '%s'\n", buffer);
        fclose(vector_file);
        exit(1);
    }

    printf("Reading a %d-column vector\n", n_cols);
    
    // Creates a sequential or parallel AIJ matrix based on number of processors in the communicator
    ierr = VecSetFromOptions(b);
    CHKERRQ(ierr);

    // set up matrix dimensions
    ierr = VecSetSizes(b, n_cols, n_cols);
    CHKERRQ(ierr);
    
    // allocate memory
    ierr = VecSetUp(b);
    CHKERRQ(ierr);

    // read in data
    printf("Reading data\n");
    for(col = 0; col < n_cols; col++){
        PetscReal value;

        fgets(buffer, 255, vector_file);
        if (sscanf(buffer, "%lf", &value) != 1)
        {
            printf("Failed to parse vector entry %d\n", col);
            printf("Line: '%s'\n", buffer);
            fclose(vector_file);
            exit(1);
        }

        ierr = VecSetValue(b, col, value, INSERT_VALUES);
        CHKERRQ(ierr);
    }

    fclose(vector_file);

    printf("Read data\n");

    // finalize data structures
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    return b;
}


int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;               /* matrix */
  Vec            x,b,u;           /* approx solution, RHS, exact solution */
  char           matrix_flname[PETSC_MAX_PATH_LEN], rhs_flname[PETSC_MAX_PATH_LEN];
  PetscBool      matrix_given=PETSC_FALSE, rhs_given=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscReal      norm;
  PetscInt       its;
  struct timespec before, after;
  PC             pc;           // pre-conditioner

  PetscInitialize(&argc, &args, (char*) 0, help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-matrix",matrix_flname,PETSC_MAX_PATH_LEN,&matrix_given);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-rhs",rhs_flname,PETSC_MAX_PATH_LEN,&rhs_given);

  if(!matrix_given) {
      printf("\n\nNeed to give -matrix parameter\n");
      exit(1);
  }

  if(!rhs_given) {
      printf("\n\nNeed to give -rhs parameter\n");
      exit(1);
  }

  printf("Loading data\n");
  A = load_my_matrix(matrix_flname);
  b = load_my_vector(rhs_flname);

  printf("Creating output vectors\n");
  ierr = MatCreateVecs(A,&x,NULL);
  CHKERRQ(ierr);
  ierr = VecSet(x,0.0);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &u);
  CHKERRQ(ierr);

  printf("Setting up KSP\n");
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);
  CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-6, 1e-6, PETSC_DEFAULT, 5000);
  CHKERRQ(ierr);
  ierr = PCCreate(PETSC_COMM_WORLD, &pc);
  CHKERRQ(ierr);
  ierr = PCSetType(pc, PCILU);
  CHKERRQ(ierr);

  printf("Solving the system\n");
  clock_gettime(CLOCK_MONOTONIC, &before);
  ierr = KSPSolve(ksp,b,x);
  clock_gettime(CLOCK_MONOTONIC, &after);
  CHKERRQ(ierr);
  printf("Done solving the system\n");

  double elapsed = 1000.0 * after.tv_sec + 1e-6 * after.tv_nsec
      - ( 1000.0 * before.tv_sec + 1e-6 * before.tv_nsec);

  printf("Time to solve system: %f ms\n", elapsed);

  ierr = KSPGetIterationNumber(ksp,&its);
  CHKERRQ(ierr);

  printf("Calculating error\n");
  ierr = MatMult(A,x,u);
  CHKERRQ(ierr);

  ierr = VecAXPY(u,-1.0,b);
  CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);
  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its);
  CHKERRQ(ierr);
  if (!PetscIsNanScalar(norm)) {
      if (norm < 1.e-12) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm < 1.e-12\n");
          CHKERRQ(ierr);
      } else {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)norm);
          CHKERRQ(ierr);
      }
  }

  ierr = MatDestroy(&A);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = VecDestroy(&x);
  CHKERRQ(ierr);
  ierr = VecDestroy(&u);
  CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = PCDestroy(&pc);
  CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  
  return 0;
}



