
/*
 * Copyright (C) 2021  Ray Kim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// #include <functional>
// #include <iostream>

// //#include <spdlog/spdlog.h>
// //#include <docopt/docopt.h>

// static constexpr auto USAGE =
//   R"(Naval Fate.

//     Usage:
//           naval_fate ship new <name>...
//           naval_fate ship <name> move <x> <y> [--speed=<kn>]
//           naval_fate ship shoot <x> <y>
//           naval_fate mine (set|remove) <x> <y> [--moored | --drifting]
//           naval_fate (-h | --help)
//           naval_fate --version
//  Options:
//           -h --help     Show this screen.
//           --version     Show version.
//           --speed=<kn>  Speed in knots [default: 10].
//           --moored      Moored (anchored) mine.
//           --drifting    Drifting mine.
// )";

// int main(int argc, const char **argv)
// {
//   // std::map<std::string, docopt::value> args = docopt::docopt(USAGE,
//   //   { std::next(argv), std::next(argv, argc) },
//   //   true,// show help if requested
//   //   "Naval Fate 2.0");// version string

//   // for (auto const &arg : args) {
//   //   std::cout << arg.first << arg.second << std::endl;
//   // }


//   // //Use the default logger (stdout, multi-threaded, colored)
//   // //spdlog::info("Hello, {}!", "World");

//   // fmt::print("Hello, from {}\n", "{fmt}");
// }

#define STATS_ENABLE_BLAZE_WRAPPERS
#include <stats.hpp>

#include <Random123/philox.h>

#include <imgui.h>
#include <imgui-SFML.h>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

int main()
{
  auto rng = r123::Philox4x32();
  (void)rng;

  sf::RenderWindow window(sf::VideoMode(640, 480), "");
  window.setVerticalSyncEnabled(true);
  ImGui::SFML::Init(window);

  sf::Color bgColor;

  blaze::DynamicMatrix<double> gamma_rvs = stats::rgamma<blaze::DynamicMatrix<double>>(100,50,3.0,2.0);



  float color[3] = { 0.f, 0.f, 0.f };

  // let's use char array as buffer, see next part
  // for instructions on using std::string with ImGui
  char windowTitle[255] = "ImGui + SFML = <3";

  window.setTitle(windowTitle);
  window.resetGLStates(); // call it if you only draw ImGui. Otherwise not needed.
  sf::Clock deltaClock;
  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      ImGui::SFML::ProcessEvent(event);

      if (event.type == sf::Event::Closed) {
	window.close();
      }
    }

    ImGui::SFML::Update(window, deltaClock.restart());

    ImGui::Begin("Sample window"); // begin window

    // Background color edit
    if (ImGui::ColorEdit3("Background color", color)) {
      // this code gets called if color value changes, so
      // the background color is upgraded automatically!
      bgColor.r = static_cast<sf::Uint8>(color[0] * 255.f);
      bgColor.g = static_cast<sf::Uint8>(color[1] * 255.f);
      bgColor.b = static_cast<sf::Uint8>(color[2] * 255.f);
    }

    // Window title text edit
    ImGui::InputText("Window title", windowTitle, 255);

    if (ImGui::Button("Update window title")) {
      // this code gets if user clicks on the button
      // yes, you could have written if(ImGui::InputText(...))
      // but I do this to show how buttons work :)
      window.setTitle(windowTitle);
    }
    ImGui::End(); // end window

    window.clear(bgColor); // fill background with color
    ImGui::SFML::Render(window);
    window.display();
  }

  ImGui::SFML::Shutdown();
}
