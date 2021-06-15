
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

#include <iostream>
#include <optional>

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

#include "ui/user_interface.hpp"
#include "ui/utils.hpp"

void
set_font()
{
  auto& io     = ImGui::GetIO();
  auto ranges  = ImVector<ImWchar>();
  auto builder = ImFontGlyphRangesBuilder();
  builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
  builder.AddRanges(io.Fonts->GetGlyphRangesKorean());
  io.Fonts->AddFontFromFileTTF(FONT_PATH, 22.0f, NULL, ranges.Data);
  ImGui::SFML::UpdateFontTexture(); 
}

int main()
{
  sf::RenderWindow window(sf::VideoMode(640, 480), "");
  window.setVerticalSyncEnabled(true);
  ImGui::SFML::Init(window, false);
  set_font();

  // let's use char array as buffer, see next part
  // for instructions on using std::string with ImGui
  char windowTitle[] = "Ultrasound Design Gallery";

  auto ui = std::optional<usdg::UserInterface>();

  window.setTitle(windowTitle);
  window.resetGLStates(); // call it if you only draw ImGui. Otherwise not needed.
  sf::Clock deltaClock;
  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      ImGui::SFML::ProcessEvent(event);
      if (event.type == sf::Event::Closed) {
	window.close();
	ui.reset();
      }
    }
    ImGui::SFML::Update(window, deltaClock.restart());
    if(!ui)
    {
      ui.emplace();
    }
    ui->state_render();
    ui->state_action();
    ui->state_transition();
    std::cout << *ui << std::endl;

    window.clear();
    ImGui::SFML::Render(window);
    window.display();
  }

  ImGui::SFML::Shutdown();
}
