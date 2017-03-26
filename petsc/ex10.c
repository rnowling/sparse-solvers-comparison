#include <stdio.h>

#include <petscksp.h>

static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -matrix <matrix_file> : matrix to use\n\n";

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

        if (sscanf(buffer, "%d %d %f", &row, &col, &value) != 3)
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

int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;               /* matrix */
  Vec            x,b,u, initialguess;           /* approx solution, RHS, exact solution */
  char           matrix_flname[PETSC_MAX_PATH_LEN];
  PetscBool      matrix_given=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscInitialize(&argc, &args, (char*) 0, help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-matrix",matrix_flname,PETSC_MAX_PATH_LEN,&matrix_given);
  CHKERRQ(ierr);

  if(!matrix_given) {
      printf("\n\nNeed to give -matrix parameter\n");
      exit(1);
  }
  
  A = load_my_matrix(matrix_flname);

  ierr = PetscFinalize();
  
  return 0;
}



