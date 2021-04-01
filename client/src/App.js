import { useState } from "react";
import { Container, CssBaseline } from "@material-ui/core";
import { ThemeProvider, createMuiTheme } from "@material-ui/core/styles";
import orange from "@material-ui/core/colors/orange";
import lightBlue from "@material-ui/core/colors/lightBlue";
import deepPurple from "@material-ui/core/colors/deepPurple";
import deepOrange from "@material-ui/core/colors/deepOrange";
import RockPaperScissor from "./components/RockPaperScissor";

const App = () => {
  const [darkState, setDarkState] = useState(
    window.localStorage.getItem("darkMode") === "true" ? true : false
  );

  const palletType = darkState ? "dark" : "light";
  const mainPrimaryColor = darkState ? orange[500] : lightBlue[500];
  const mainSecondaryColor = darkState ? deepOrange[900] : deepPurple[500];

  const darkTheme = createMuiTheme({
    palette: {
      type: palletType,
      primary: {
        main: mainPrimaryColor,
      },
      secondary: {
        main: mainSecondaryColor,
      },
    },
  });

  const handleThemeChange = () => {
    const preference = darkState;
    setDarkState(!darkState);
    window.localStorage.setItem("darkMode", !preference);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container>
        <RockPaperScissor />
      </Container>
    </ThemeProvider>
  );
};

export default App;
