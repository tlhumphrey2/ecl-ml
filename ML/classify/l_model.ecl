 IMPORT ML;
IMPORT * FROM $;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

SHARED l_model := RECORD
  Types.t_RecordId    id := 0;      // A record-id - allows a model to have an ordered sequence of results
  Types.t_FieldNumber number;       // A reference to a feature (or field) in the independants
  Types.t_Discrete    class_number; // The field number of the dependant variable
  REAL8 w;
END;
