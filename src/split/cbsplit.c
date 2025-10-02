/*-------------------------------------------------------------------*/
/* CBSPLIT.C      Timo Kaukoranta                                    */
/*                                                                   */
/*                                                                   */
/* - Splitting method (13.11.00)                                     */
/* 28.2.07: SavePartition added                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/

#define ProgName "CBSPLIT"
#define VersionNumber "V.0.38"
#define LastUpdated "28.2.07" /* PF */

/* ----------------------------------------------------------------- */

#include <string.h>
#ifdef UNIX
#include <sys/time.h>
#endif

#include <time.h>
#include <limits.h>

#define FACTFILE "cbsplit.fac"
#define BATCH

#include "parametr.h"

#include "file.h"
#include "memctrl.h"
#include "random.h"
#include "sa.h"
#include "sortcb.h"

#include "split.h"
#include "interfc.h" 

// Add function declaration for CentroidIndex
int CentroidIndex(CODEBOOK* CB1, CODEBOOK* CB2);

/*--------------------  Basic type definitions -----------------------*/

/* None. */

/*-----------------------------  M i s c  ----------------------------*/

static int ResultCBSize = 0;

/* =================  T I M E   F U N C T I O N S  ================= */

/* Use 'gettimeofday' for more accurate measure of time. */

long SetWatch(long *start) {
  time_t tmp;
  return (*start = (long)time(&tmp));
}

/* ----------------------------------------------------------------- */

void PrintTime(long start) {
  time_t tmp;
  long elapsed = (long)time(&tmp) - start;

  switch (Value(QuietLevel)) {
  case 0:
    break;
  case 1:
    printf("%6li ", elapsed);
    break;
  default:
    printf("Time:%6li\n", elapsed);
    break;
  }
}

/*============================  P R I N T  ===============================*/

void PrintInfo(void) {
  printf("%s %s %s\n", ProgName, VersionNumber, LastUpdated);
  printf("This program generates a codebook by SPLIT method.\n");
  printf("Usage: %s [%coption] <training-set> [initial-codebook] <result-codebook>\n", ProgName,
         OPTION_SYMBOL);
  printf("For example: %s set_of_4 gla glasplit\n", ProgName);
  printf("\n  Options:\n");
  PrintOptions();
  printf("%s %s %s\n", ProgName, VersionNumber, LastUpdated);
}

/*-------------------------------------------------------------------*/

static void PrintOperatingInfo(char TSetName[], char InitialCBName[], char CBName[]) {
  if (Value(QuietLevel) >= 2) {
    printf("\n%s %s %s\n", ProgName, VersionNumber, LastUpdated);
    printf("Training Set:     %s\n", TSetName);
    printf("Initial Codebook: %s\n",
           (InitialCBName[0] == 0x00 ? "Centroid of the TRAINING SET" : InitialCBName));
    printf("Result Codebook:  %s\n", CBName);
    printf("Result CB size:   %u\n", ResultCBSize);
    PrintSelectedOptions();
    printf("\n");
  }
}

/*=========================  CODEBOOK HANDLING  ============================*/

static void GenerateInitialCodebook(char *InitialCBName, TRAININGSET *TS, CODEBOOK *CB,
                                    PARTITIONING *P) {
  if (InitialCBName[0] != 0x00) /* Initial CB is given. */
  {
    ReadCodebook(InitialCBName, CB);
    if (Value(CodebookSizeChange) > 0) {
      ResultCBSize = BookSize(CB) + Value(CodebookSizeChange);
    } else {
      ResultCBSize = Value(CodebookSize);
    }

    IncreaseCodebookSize(CB, ResultCBSize);
    CreateNewPartitioning(P, TS, ResultCBSize);
    GenerateOptimalPartitioning(TS, CB, P);

    AddGenerationMethod(CB, "SPLIT");
  } else /* No initial CB */
  {
    /* Use the centroid of the TS as a initial CB. */
    if (Value(CodebookSizeChange) > 0) {
      ResultCBSize = 1 + Value(CodebookSizeChange);
    } else {
      ResultCBSize = Value(CodebookSize);
    }

    CreateNewCodebook(CB, 1, TS);
    CreateNewPartitioning(P, TS, 1);

    /* Just centroid of TS. */
    PartitionCentroid(P, 0, &Node(CB, 0));

    AddGenerationMethod(CB, "SPLIT");
  }
}

/* ----------------------------------------------------------------- */

static void SaveSolution(char *CBName, char *PAName, TRAININGSET *TS, CODEBOOK *CB, PARTITIONING *P)

{
  int ndups;

  ndups = DuplicatesInCodebook(CB);
  if ((ndups > 0) && (Value(QuietLevel) >= 2)) {
    printf("WARNING: Final CB contains %i duplicates.\n", ndups);
  }
  SortCodebook(CB, FREQ_DESCENDING);
  WriteCodebook(CBName, CB, Value(OverWrite));
  if (Value(SavePartition)) {
    WritePartitioning(PAName, P, TS, Value(OverWrite));
  }
}

/* ================================================================= */

static void Initialize(TRAININGSET *TS, CODEBOOK *CB, SASchedule *SAS) {
  int ndups;

  if ((ndups = DuplicatesInCodebook(CB)) > 0) {
    if (Value(QuietLevel) >= 1) {
      printf("ERROR: Init. CB does contain %i duplicates.\n", ndups);
      exit(-1);
    }
  }

  SetSplitParameters(Value(QuietLevel), Value(PartitionErrorType), Value(NewVectors),
                     Value(NewVectorsHeuristically), Value(Hyperplane), Value(HyperplanePivot),
                     Value(HyperplaneLine), Value(PartitionRemapping), Value(LocalGLA),
                     Value(LocalGLAIterations), Value(GLAIterations));
  InitializeSASchedule(
      Value(TemperatureDistribution), Value(TemperatureFunction), Value(TemperatureChange),
      (float)Value(TemperatureInitial) * (float)(CB->MaxValue) / 100.0,
      (Value(VectorRandomizing) == RandomCentroid || Value(VectorRandomizing) == RandomBoth),
      (Value(VectorRandomizing) == RandomTrainingSet || Value(VectorRandomizing) == RandomBoth),
      SAS);
}

/* ================================================================= */

static void ShutUp(TRAININGSET *TS, CODEBOOK *CB, PARTITIONING *P) {
  FreeCodebook(TS);
  FreeCodebook(CB);
  FreePartitioning(P);
}

/*===========================  M A I N  ==============================*/

int main(int argc, char *argv[]) {
  CODEBOOK CB;
  TRAININGSET TS;
  PARTITIONING P;
  char TSetName[MAXFILENAME] = {0x00};
  char InitialCBName[MAXFILENAME] = {0x00};
  char CBName[MAXFILENAME] = {0x00};
  char PAName[MAXFILENAME] = {0x00};
  char ValidationName[MAXFILENAME] = {0x00};
  SASchedule SAS;
  double FinalError;
  long watch;

  // Try parsing with 4 parameters first
  ParameterInfo paraminfo4[4] = {{TSetName, FormatNameTS, 0, INFILE},
                                 {InitialCBName, FormatNameCB, 1, INFILE},
                                 {CBName, FormatNameCB, 0, OUTFILE},
                                 {ValidationName, FormatNameCB, 0, INFILE}};

  // Try parsing with 3 parameters as fallback
  ParameterInfo paraminfo3[3] = {{TSetName, FormatNameTS, 0, INFILE},
                                 {InitialCBName, FormatNameCB, 1, INFILE},
                                 {CBName, FormatNameCB, 0, OUTFILE}};

  // Count non-option arguments to determine which parser to use
  int fileArgCount = 0;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-') {
      fileArgCount++;
    }
  }

  // Use appropriate parameter parser based on file argument count
  if (fileArgCount >= 3) {
    ParseParameters(argc, argv, 4, paraminfo4);
  } else {
    ParseParameters(argc, argv, 3, paraminfo3);
  }

  PickFileName(CBName, PAName);
  CheckFileName(PAName, FormatNamePA);

  // Debug: Print all command line arguments
  if (Value(QuietLevel) >= 2) {
    printf("DEBUG: Command line arguments:\n");
    for (int i = 0; i < argc; i++) {
      printf("  argv[%d] = '%s'\n", i, argv[i]);
    }
    printf("DEBUG: File argument count = %d\n", fileArgCount);
    printf("DEBUG: ValidationName = '%s'\n", ValidationName);
  }

  initrandom(Value(RandomSeed));
  ReadTrainingSet(TSetName, &TS);
  SetWatch(&watch);
  GenerateInitialCodebook(InitialCBName, &TS, &CB, &P);
  Initialize(&TS, &CB, &SAS);
  PrintOperatingInfo(TSetName, InitialCBName, CBName);

  Split(&TS, &CB, &SAS, &P, ResultCBSize);

  PrintTime(watch);

  FinalError = AverageErrorCBFast(&TS, &CB, &P, MSE);
  if (Value(QuietLevel) >= 2) {
    printf("Final dist. = %9.4f\n", PrintableError(FinalError, &CB));
  } else {
    printf("%9.4f ", PrintableError(FinalError, &CB));
  }

  // Calculate Centroid Index if ground truth is provided
  if (ValidationName[0] != 0x00) {
    CODEBOOK GT_CB;
    int ci;
    
    if (Value(QuietLevel) >= 2) {
      printf("Reading validation codebook: %s\n", ValidationName);
    }
    
    ReadCodebook(ValidationName, &GT_CB);
    ci = CentroidIndex(&CB, &GT_CB);
    
    if (Value(QuietLevel) >= 2) {
      printf("Centroid Index = %d\n", ci);
    } else {
      printf("%d ", ci);
    }
    
    FreeCodebook(&GT_CB);
  } else {
    if (Value(QuietLevel) >= 2) {
      printf("No validation file provided\n");
    }
  }

  if (TS.InputFormat == TXT) {
    SaveCB2TXT(&CB, CBName, TS.MinMax, NULL, NULL);
  } else {
    WriteCodebook(CBName, &CB, Value(OverWrite));
  }

  ShutUp(&TS, &CB, &P);
  checkmemory();

  return (0);
}
