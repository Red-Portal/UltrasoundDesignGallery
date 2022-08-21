
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

#include <iostream>
#include <optional>

#include <unistd.h>

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
  ImGui::StyleColorsLight();

  auto& io = ImGui::GetIO();
  io.ConfigFlags          |= ImGuiConfigFlags_NavEnableKeyboard;
  io.FontAllowUserScaling  = true;

  char windowTitle[] = "Ultrasound Design Gallery";
  auto ui = std::optional<usdg::UserInterface>();

  window.setTitle(windowTitle);
  window.resetGLStates();
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
    ImGui::SetNextWindowPos(ImVec2(0,0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("main",
		 nullptr,
		 ImGuiWindowFlags_NoTitleBar
		 | ImGuiWindowFlags_NoResize
		 | ImGuiWindowFlags_NoMove
		 | ImGuiWindowFlags_NoScrollbar
		 | ImGuiWindowFlags_NoScrollWithMouse
		 | ImGuiWindowFlags_NoBringToFrontOnFocus);
    {
      ui->state_render();
      ui->state_action();
      ui->state_transition();
    }
    ImGui::End();

    //std::cout << *ui << std::endl;

    window.clear();
    ImGui::SFML::Render(window);
    window.display();
  }

  ImGui::SFML::Shutdown();
}
