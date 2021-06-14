
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

#include <string>

#include <imgui.h>
#include <imgui-SFML.h>
#include <portable-file-dialogs.h>

#include "user_interface.hpp"

using namespace std::literals::string_literals;

namespace usdg
{
  void
  UserInterface::
  render_menubar()
  {
    if (ImGui::BeginMainMenuBar())
    {
      if (ImGui::BeginMenu("File"))
      {
	if (ImGui::MenuItem("Open File"))
	{
	  if(_video_player)
	    _video_player.reset();
	  auto result = pfd::open_file(
	    "Select File"s,
	    {"Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.gif *.psd *.hdr *.pic",
	     // "Video Files"s, "*.mp4 *.wav", automate this by ffmpeg -demuxers
	    }).result();

	  if(!result.empty())
	    _video_player.emplace(result[0]);
	}
	//ShowExampleMenuFile();
	ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Edit"))
      {
	if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
	if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
	ImGui::Separator();
	if (ImGui::MenuItem("Cut", "CTRL+X")) {}
	if (ImGui::MenuItem("Copy", "CTRL+C")) {}
	if (ImGui::MenuItem("Paste", "CTRL+V")) {}
	ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
  }

  void
  UserInterface::
  render()
  {
    render_menubar();
    _linesearch.render();
    if(_video_player)
    {
      double param = _linesearch.selected_parameter();
      std::cout << param << std::endl;
      //_video_player->update_parameter(param);
      _video_player->render();
    }
  }
}
