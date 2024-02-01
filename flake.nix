{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {self, nixpkgs, ... }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
      (system: f system (import nixpkgs {
        inherit system;
        config.allowUnfree=true;
        config.cudaSupport=true;
        config.cudaCapabilities = [ "8.6" ];
      }));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let
            kt-legacy = pkgs.python3.pkgs.buildPythonPackage rec {
              pname = "kt-legacy";
              version = "1.0.4";
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-qUES5CpQ58w6rTHzKHqjhMI1VeoUMsVbWCOFLgnnBs8=";
              };
              nativeBuildInputs = with pkgs.python3Packages; [];
              doCheck=false;
              meta = {
                homepage = "https://keras-team.github.io/keras-tuner";
                description = "hyperparameter tuning for TensorFlow/Keras";
              };
            };


            keras-tuner = pkgs.python3.pkgs.buildPythonPackage rec {
              pname = "keras-tuner";
              version = "1.1.3";
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-KW7bTo/buFY8AUUnPqlPuIiZQoQaH87xORrglFVHfXA=";
              };
              propagatedBuildInputs = with pkgs.python3Packages; [tensorflow-tensorboard packaging numpy requests ipython kt-legacy scipy];
              doCheck=false;
              meta = {
                homepage = "https://keras-team.github.io/keras-tuner";
                description = "hyperparameter tuning for TensorFlow/Keras";
              };
            };
            python = pkgs.python3.withPackages (p: with p;[numpy
              matplotlib tensorflow tqdm keras keras-tuner]);
        in pkgs.mkShell {
            buildInputs = [
              python
            ];
          }
        );
    };
}
