%% hydro data
hydro = struct();
hydro = readCAPYTAINE(hydro,'bem.nc','hs');
hydro = radiationIRF(hydro,60,[],[],[],[]);
% hydro = radiationIRFSS(hydro,[],[]);
hydro = excitationIRF(hydro,157,[],[],[],[]);

disp(hydro)

% Mod
% hydro.Khs = zeros(6,6,3);
% hydro.A = zeros(6,6,10);
% hydro.Ainf = hydro.Ainf * 0;
% hydro.B = zeros(18,18,10);

writeBEMIOH5(hydro)
% plotBEMIO(hydro)