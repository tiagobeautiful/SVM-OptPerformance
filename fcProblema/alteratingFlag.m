function [STR] = alteratingFlag(method,STR)

switch method
    case 'Knapsack_SVM'
        STR.runKnapsack_SVM       = 1;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Knapsack_ProjGrad'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 1;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'ProjGrad_cauchy'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 1;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'ProjGrad_newton'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 1;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'SPG_exactSearch'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 1;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'SPG_augLag'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 1;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'SPG_augLag_rho'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 1;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Filter_qp'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 1;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Filter_AL'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 1;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Libsvm'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 1;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Quadprog'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 1;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;

    case 'Quadprog_reg'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 1;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'Liblinear'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 1;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 0;
        
    case 'SMO'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 1;
        STR.runMyLibsvm           = 0;
    
    case 'My_LIBSVM'
        STR.runKnapsack_SVM       = 0;
        STR.runKnapsack_ProjGrad  = 0;
        STR.runProjGrad_cauchy    = 0;
        STR.runProjGrad_newton    = 0;
        STR.runSPG_exactSearch    = 0;
        STR.runSPG_augLag         = 0;
        STR.runSPG_augLag_rho     = 0;
        STR.runFilter_qp          = 0;
        STR.runFilter_AL          = 0;
        STR.runLibsvm             = 0;
        STR.runQuadprog           = 0;
        STR.runQuadprog_reg       = 0;
        STR.runLiblinear          = 0;
        STR.runSMO                = 0;
        STR.runMyLibsvm           = 1;
        
end




end