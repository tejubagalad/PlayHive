import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import "./App.css";

const App = () => {
  const getBaseUrl = () => {
    const host = window.location.hostname;
    if (host.includes("cloudapp.azure.com")) {
      return "http://playhive.southindia.cloudapp.azure.com";
    }
    if (host.includes("gamehub.local") || host.includes("localhost")) {
      return "";
    }
    return "";
  };

  const getGameUrl = (game) => {
    const baseUrl = getBaseUrl();
    if (game === "2048") return `${baseUrl}/2048/`;
    if (game === "snake") return `${baseUrl}/snake/`;
    return "/";
  };

  const redirectToGame = (game) => {
    window.location.href = getGameUrl(game);
  };

  return (
    <div className="bg-dark text-light min-vh-100 d-flex flex-column justify-content-between">
      {/* Navbar */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-black shadow-sm w-100">
        <div className="container justify-content-center">
          <span className="navbar-brand fw-bold fs-3 text-success">
            🎮 PlayHive
          </span>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container-fluid text-center flex-grow-1 d-flex flex-column justify-content-center align-items-center">
        <h1 className="fw-bold text-success mb-4 display-6">
          Welcome to the PlayHive!
        </h1>
        <p className="text-secondary mb-4">
           CHOOSE YOUR GAME AND HAVE FUN!!!!
        </p>

        <div className="row justify-content-center g-3 w-100 px-3">
          <div className="col-12 col-sm-6 col-md-4">
            <button
              onClick={() => redirectToGame("2048")}
              className="btn btn-primary w-100 py-3 fs-5"
            >
              <i className="bi bi-grid-3x3-gap-fill me-2"></i>
              Play 2048
            </button>
          </div>

          <div className="col-12 col-sm-6 col-md-4">
            <button
              onClick={() => redirectToGame("snake")}
              className="btn btn-success w-100 py-3 fs-5"
            >
              <i className="bi bi-controller me-2"></i>
              Play Snake
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black text-center text-muted py-3 small w-100">
        © {new Date().getFullYear()} PlayHive | Built with 💙 React + Bootstrap
      </footer>
    </div>
  );
};

export default App;
